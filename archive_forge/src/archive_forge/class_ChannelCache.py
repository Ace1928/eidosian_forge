import collections
import datetime
import logging
import os
import threading
from typing import (
import grpc
from grpc.experimental import experimental_api
class ChannelCache:
    _singleton = None
    _lock: threading.RLock = threading.RLock()
    _condition: threading.Condition = threading.Condition(lock=_lock)
    _eviction_ready: threading.Event = threading.Event()
    _mapping: Dict[CacheKey, Tuple[grpc.Channel, datetime.datetime]]
    _eviction_thread: threading.Thread

    def __init__(self):
        self._mapping = collections.OrderedDict()
        self._eviction_thread = threading.Thread(target=ChannelCache._perform_evictions, daemon=True)
        self._eviction_thread.start()

    @staticmethod
    def get():
        with ChannelCache._lock:
            if ChannelCache._singleton is None:
                ChannelCache._singleton = ChannelCache()
        ChannelCache._eviction_ready.wait()
        return ChannelCache._singleton

    def _evict_locked(self, key: CacheKey):
        channel, _ = self._mapping.pop(key)
        _LOGGER.debug('Evicting channel %s with configuration %s.', channel, key)
        channel.close()
        del channel

    @staticmethod
    def _perform_evictions():
        while True:
            with ChannelCache._lock:
                ChannelCache._eviction_ready.set()
                if not ChannelCache._singleton._mapping:
                    ChannelCache._condition.wait()
                elif len(ChannelCache._singleton._mapping) > _MAXIMUM_CHANNELS:
                    key = next(iter(ChannelCache._singleton._mapping.keys()))
                    ChannelCache._singleton._evict_locked(key)
                else:
                    key, (_, eviction_time) = next(iter(ChannelCache._singleton._mapping.items()))
                    now = datetime.datetime.now()
                    if eviction_time <= now:
                        ChannelCache._singleton._evict_locked(key)
                        continue
                    else:
                        time_to_eviction = (eviction_time - now).total_seconds()
                        ChannelCache._condition.wait(timeout=time_to_eviction)

    def get_channel(self, target: str, options: Sequence[Tuple[str, str]], channel_credentials: Optional[grpc.ChannelCredentials], insecure: bool, compression: Optional[grpc.Compression]) -> grpc.Channel:
        if insecure and channel_credentials:
            raise ValueError('The insecure option is mutually exclusive with ' + 'the channel_credentials option. Please use one ' + 'or the other.')
        if insecure:
            channel_credentials = grpc.experimental.insecure_channel_credentials()
        elif channel_credentials is None:
            _LOGGER.debug('Defaulting to SSL channel credentials.')
            channel_credentials = grpc.ssl_channel_credentials()
        key = (target, options, channel_credentials, compression)
        with self._lock:
            channel_data = self._mapping.get(key, None)
            if channel_data is not None:
                channel = channel_data[0]
                self._mapping.pop(key)
                self._mapping[key] = (channel, datetime.datetime.now() + _EVICTION_PERIOD)
                return channel
            else:
                channel = _create_channel(target, options, channel_credentials, compression)
                self._mapping[key] = (channel, datetime.datetime.now() + _EVICTION_PERIOD)
                if len(self._mapping) == 1 or len(self._mapping) >= _MAXIMUM_CHANNELS:
                    self._condition.notify()
                return channel

    def _test_only_channel_count(self) -> int:
        with self._lock:
            return len(self._mapping)