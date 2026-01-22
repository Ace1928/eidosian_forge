import time
import threading
class BandwidthLimitedStream(object):

    def __init__(self, fileobj, leaky_bucket, transfer_coordinator, time_utils=None, bytes_threshold=256 * 1024):
        """Limits bandwidth for reads on a wrapped stream

        :type fileobj: file-like object
        :param fileobj: The file like object to wrap

        :type leaky_bucket: LeakyBucket
        :param leaky_bucket: The leaky bucket to use to throttle reads on
            the stream

        :type transfer_coordinator: s3transfer.futures.TransferCoordinator
        param transfer_coordinator: The coordinator for the general transfer
            that the wrapped stream is a part of

        :type time_utils: TimeUtils
        :param time_utils: The time utility to use for interacting with time
        """
        self._fileobj = fileobj
        self._leaky_bucket = leaky_bucket
        self._transfer_coordinator = transfer_coordinator
        self._time_utils = time_utils
        if time_utils is None:
            self._time_utils = TimeUtils()
        self._bandwidth_limiting_enabled = True
        self._request_token = RequestToken()
        self._bytes_seen = 0
        self._bytes_threshold = bytes_threshold

    def enable_bandwidth_limiting(self):
        """Enable bandwidth limiting on reads to the stream"""
        self._bandwidth_limiting_enabled = True

    def disable_bandwidth_limiting(self):
        """Disable bandwidth limiting on reads to the stream"""
        self._bandwidth_limiting_enabled = False

    def read(self, amount):
        """Read a specified amount

        Reads will only be throttled if bandwidth limiting is enabled.
        """
        if not self._bandwidth_limiting_enabled:
            return self._fileobj.read(amount)
        self._bytes_seen += amount
        if self._bytes_seen < self._bytes_threshold:
            return self._fileobj.read(amount)
        self._consume_through_leaky_bucket()
        return self._fileobj.read(amount)

    def _consume_through_leaky_bucket(self):
        while not self._transfer_coordinator.exception:
            try:
                self._leaky_bucket.consume(self._bytes_seen, self._request_token)
                self._bytes_seen = 0
                return
            except RequestExceededException as e:
                self._time_utils.sleep(e.retry_time)
        else:
            raise self._transfer_coordinator.exception

    def signal_transferring(self):
        """Signal that data being read is being transferred to S3"""
        self.enable_bandwidth_limiting()

    def signal_not_transferring(self):
        """Signal that data being read is not being transferred to S3"""
        self.disable_bandwidth_limiting()

    def seek(self, where):
        self._fileobj.seek(where)

    def tell(self):
        return self._fileobj.tell()

    def close(self):
        if self._bandwidth_limiting_enabled and self._bytes_seen:
            self._consume_through_leaky_bucket()
        self._fileobj.close()

    def __enter__(self):
        return self

    def __exit__(self, *args, **kwargs):
        self.close()