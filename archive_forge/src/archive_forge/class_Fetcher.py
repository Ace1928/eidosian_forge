import asyncio
import collections
import logging
import random
import time
from itertools import chain
import async_timeout
import aiokafka.errors as Errors
from aiokafka.errors import (
from aiokafka.protocol.offset import OffsetRequest
from aiokafka.protocol.fetch import FetchRequest
from aiokafka.record.memory_records import MemoryRecords
from aiokafka.record.control_record import ControlRecord, ABORT_MARKER
from aiokafka.structs import OffsetAndTimestamp, TopicPartition, ConsumerRecord
from aiokafka.util import create_future, create_task
class Fetcher:
    """Initialize a Kafka Message Fetcher.

    Parameters:
        client (AIOKafkaClient): kafka client
        subscription (SubscriptionState): instance of SubscriptionState
            located in aiokafka.consumer.subscription_state
        key_deserializer (callable): Any callable that takes a
            raw message key and returns a deserialized key.
        value_deserializer (callable, optional): Any callable that takes a
            raw message value and returns a deserialized value.
        fetch_min_bytes (int): Minimum amount of data the server should
            return for a fetch request, otherwise wait up to
            fetch_max_wait_ms for more data to accumulate. Default: 1.
        fetch_max_bytes (int): The maximum amount of data the server should
            return for a fetch request. This is not an absolute maximum, if
            the first message in the first non-empty partition of the fetch
            is larger than this value, the message will still be returned
            to ensure that the consumer can make progress. NOTE: consumer
            performs fetches to multiple brokers in parallel so memory
            usage will depend on the number of brokers containing
            partitions for the topic.
            Supported Kafka version >= 0.10.1.0. Default: 52428800 (50 Mb).
        fetch_max_wait_ms (int): The maximum amount of time in milliseconds
            the server will block before answering the fetch request if
            there isn't sufficient data to immediately satisfy the
            requirement given by fetch_min_bytes. Default: 500.
        max_partition_fetch_bytes (int): The maximum amount of data
            per-partition the server will return. The maximum total memory
            used for a request = #partitions * max_partition_fetch_bytes.
            This size must be at least as large as the maximum message size
            the server allows or else it is possible for the producer to
            send messages larger than the consumer can fetch. If that
            happens, the consumer can get stuck trying to fetch a large
            message on a certain partition. Default: 1048576.
        check_crcs (bool): Automatically check the CRC32 of the records
            consumed. This ensures no on-the-wire or on-disk corruption to
            the messages occurred. This check adds some overhead, so it may
            be disabled in cases seeking extreme performance. Default: True
        fetcher_timeout (float): Maximum polling interval in the background
            fetching routine. Default: 0.2
        prefetch_backoff (float): number of seconds to wait until
            consumption of partition is paused. Paused partitions will not
            request new data from Kafka server (will not be included in
            next poll request).
        auto_offset_reset (str): A policy for resetting offsets on
            OffsetOutOfRange errors: 'earliest' will move to the oldest
            available message, 'latest' will move to the most recent. Any
            ofther value will raise the exception. Default: 'latest'.
        isolation_level (str): Controls how to read messages written
            transactionally. See consumer description.
    """

    def __init__(self, client, subscriptions, *, key_deserializer=None, value_deserializer=None, fetch_min_bytes=1, fetch_max_bytes=52428800, fetch_max_wait_ms=500, max_partition_fetch_bytes=1048576, check_crcs=True, fetcher_timeout=0.2, prefetch_backoff=0.1, retry_backoff_ms=100, auto_offset_reset='latest', isolation_level='read_uncommitted'):
        self._client = client
        self._loop = client._loop
        self._key_deserializer = key_deserializer
        self._value_deserializer = value_deserializer
        self._fetch_min_bytes = fetch_min_bytes
        self._fetch_max_bytes = fetch_max_bytes
        self._fetch_max_wait_ms = fetch_max_wait_ms
        self._max_partition_fetch_bytes = max_partition_fetch_bytes
        self._check_crcs = check_crcs
        self._fetcher_timeout = fetcher_timeout
        self._prefetch_backoff = prefetch_backoff
        self._retry_backoff = retry_backoff_ms / 1000
        self._subscriptions = subscriptions
        self._default_reset_strategy = OffsetResetStrategy.from_str(auto_offset_reset)
        if isolation_level == 'read_uncommitted':
            self._isolation_level = READ_UNCOMMITTED
        elif isolation_level == 'read_committed':
            self._isolation_level = READ_COMMITTED
        else:
            raise ValueError(f'Incorrect isolation level {isolation_level}')
        self._records = collections.OrderedDict()
        self._in_flight = set()
        self._pending_tasks = set()
        self._wait_consume_future = None
        self._fetch_waiters = set()
        self._subscriptions.register_fetch_waiters(self._fetch_waiters)
        if client.api_version >= (0, 11):
            req_version = 4
        elif client.api_version >= (0, 10, 1):
            req_version = 3
        elif client.api_version >= (0, 10):
            req_version = 2
        else:
            req_version = 1
        self._fetch_request_class = FetchRequest[req_version]
        self._fetch_task = create_task(self._fetch_requests_routine())
        self._closed = False

    async def close(self):
        self._closed = True
        self._fetch_task.cancel()
        try:
            await self._fetch_task
        except asyncio.CancelledError:
            pass
        for waiter in self._fetch_waiters:
            self._notify(waiter)
        for x in self._pending_tasks:
            x.cancel()
            await x

    def _notify(self, future):
        if future is not None and (not future.done()):
            future.set_result(None)

    def _create_fetch_waiter(self):
        fut = self._loop.create_future()
        self._fetch_waiters.add(fut)
        fut.add_done_callback(lambda f, waiters=self._fetch_waiters: waiters.remove(f))
        return fut

    @property
    def error_future(self):
        return self._fetch_task

    async def _fetch_requests_routine(self):
        """ Implements a background task to populate internal fetch queue
        ``self._records`` with prefetched messages. This helps isolate the
        ``getall/getone`` calls from actual calls to broker. This way we don't
        need to think of what happens if user calls get in 2 tasks, etc.

            The loop is quite complicated due to a large set of events that
        can allow new fetches to be send. Those include previous fetches,
        offset resets, metadata updates to discover new leaders for partitions,
        data consumed for partition.

            Previously the offset reset was performed separately, but it did
        not perform too reliably. In ``kafka-python`` and Java client the reset
        is perform in ``poll()`` before each fetch, which works good for sync
        systems. But with ``aiokafka`` the user can actually break such
        behaviour quite easily by performing actions from different tasks.
        """
        try:
            assignment = None

            def start_pending_task(coro, node_id, self=self):
                task = create_task(coro)
                self._pending_tasks.add(task)
                self._in_flight.add(node_id)

                def on_done(fut, self=self):
                    self._in_flight.discard(node_id)
                task.add_done_callback(on_done)
            while True:
                if assignment is None or not assignment.active:
                    for task in self._pending_tasks:
                        if not task.done():
                            task.cancel()
                        await task
                    self._pending_tasks.clear()
                    self._records.clear()
                    subscription = self._subscriptions.subscription
                    if subscription is None or subscription.assignment is None:
                        try:
                            waiter = self._subscriptions.wait_for_assignment()
                            await waiter
                        except Errors.KafkaError:
                            continue
                    assignment = self._subscriptions.subscription.assignment
                assert assignment is not None and assignment.active
                self._wait_consume_future = create_future()
                fetch_requests, reset_requests, timeout, invalid_metadata, resume_futures = self._get_actions_per_node(assignment)
                for node_id, request in fetch_requests:
                    start_pending_task(self._proc_fetch_request(assignment, node_id, request), node_id=node_id)
                for node_id, tps in reset_requests.items():
                    start_pending_task(self._update_fetch_positions(assignment, node_id, tps), node_id=node_id)
                other_futs = [self._wait_consume_future, assignment.unassign_future]
                if invalid_metadata:
                    fut = self._client.force_metadata_update()
                    other_futs.append(fut)
                done_set, _ = await asyncio.wait(set(chain(self._pending_tasks, other_futs, resume_futures)), timeout=timeout, return_when=asyncio.FIRST_COMPLETED)
                done_pending = self._pending_tasks.intersection(done_set)
                if done_pending:
                    has_new_data = any((fut.result() for fut in done_pending))
                    if has_new_data:
                        for waiter in self._fetch_waiters:
                            self._notify(waiter)
                    self._pending_tasks -= done_pending
        except asyncio.CancelledError:
            pass
        except Exception:
            log.error('Unexpected error in fetcher routine', exc_info=True)
            raise Errors.KafkaError('Unexpected error during data retrieval')

    def _get_actions_per_node(self, assignment):
        """ For each assigned partition determine the action needed to be
        performed and group those by leader node id.
        """
        fetchable = collections.defaultdict(list)
        awaiting_reset = collections.defaultdict(list)
        backoff_by_nodes = collections.defaultdict(list)
        resume_futures = []
        invalid_metadata = False
        for tp in assignment.tps:
            tp_state = assignment.state_value(tp)
            node_id = self._client.cluster.leader_for_partition(tp)
            backoff = 0
            if tp in self._records:
                record = self._records[tp]
                backoff = record.calculate_backoff()
                if backoff:
                    backoff_by_nodes[node_id].append(backoff)
            elif node_id in self._in_flight:
                continue
            elif node_id is None or node_id == -1:
                log.debug('No leader found for partition %s. Waiting metadata update', tp)
                invalid_metadata = True
            elif not tp_state.has_valid_position:
                awaiting_reset[node_id].append(tp)
            elif tp_state.paused:
                resume_futures.append(tp_state.resume_fut)
            else:
                position = tp_state.position
                fetchable[node_id].append((tp, position))
                log.debug('Adding fetch request for partition %s at offset %d', tp, position)
        fetch_requests = []
        for node_id, partition_data in fetchable.items():
            if node_id in backoff_by_nodes:
                continue
            if node_id in awaiting_reset:
                continue
            random.shuffle(partition_data)
            by_topics = collections.defaultdict(list)
            for tp, position in partition_data:
                by_topics[tp.topic].append((tp.partition, position, self._max_partition_fetch_bytes))
            klass = self._fetch_request_class
            if klass.API_VERSION > 3:
                req = klass(-1, self._fetch_max_wait_ms, self._fetch_min_bytes, self._fetch_max_bytes, self._isolation_level, list(by_topics.items()))
            elif klass.API_VERSION == 3:
                req = klass(-1, self._fetch_max_wait_ms, self._fetch_min_bytes, self._fetch_max_bytes, list(by_topics.items()))
            else:
                req = klass(-1, self._fetch_max_wait_ms, self._fetch_min_bytes, list(by_topics.items()))
            fetch_requests.append((node_id, req))
        if backoff_by_nodes:
            backoff = min(map(max, backoff_by_nodes.values()))
        else:
            backoff = self._fetcher_timeout
        return (fetch_requests, awaiting_reset, backoff, invalid_metadata, resume_futures)

    async def _proc_fetch_request(self, assignment, node_id, request):
        needs_wakeup = False
        try:
            response = await self._client.send(node_id, request)
        except Errors.KafkaError as err:
            log.error('Failed fetch messages from %s: %s', node_id, err)
            await asyncio.sleep(self._retry_backoff)
            return False
        except asyncio.CancelledError:
            return False
        if not assignment.active:
            log.debug('Discarding fetch response since the assignment changed during fetch')
            return False
        fetch_offsets = {}
        for topic, partitions in request.topics:
            for partition, offset, _ in partitions:
                fetch_offsets[TopicPartition(topic, partition)] = offset
        now_ms = int(1000 * time.time())
        for topic, partitions in response.topics:
            for partition, error_code, highwater, *part_data in partitions:
                tp = TopicPartition(topic, partition)
                error_type = Errors.for_code(error_code)
                fetch_offset = fetch_offsets[tp]
                tp_state = assignment.state_value(tp)
                if not tp_state.has_valid_position or tp_state.position != fetch_offset:
                    log.debug('Discarding fetch response for partition %s since its offset %s does not match the current position', tp, fetch_offset)
                    continue
                if error_type is Errors.NoError:
                    if request.API_VERSION >= 4:
                        aborted_transactions = part_data[-2]
                        lso = part_data[-3]
                    else:
                        aborted_transactions = None
                        lso = None
                    tp_state.highwater = highwater
                    tp_state.lso = lso
                    tp_state.timestamp = now_ms
                    records = MemoryRecords(part_data[-1])
                    if records.has_next():
                        log.debug('Adding fetched record for partition %s with offset %d to buffered record list', tp, fetch_offset)
                        partition_records = PartitionRecords(tp, records, aborted_transactions, fetch_offset, self._key_deserializer, self._value_deserializer, self._check_crcs, self._isolation_level)
                        self._records[tp] = FetchResult(tp, partition_records=partition_records, assignment=assignment, backoff=self._prefetch_backoff)
                        needs_wakeup = True
                    elif records.size_in_bytes() > 0:
                        err = RecordTooLargeError('There are some messages at [Partition=Offset]: %s=%s whose size is larger than the fetch size %s and hence cannot be ever returned. Increase the fetch size, or decrease the maximum message size the broker will allow.', tp, fetch_offset, self._max_partition_fetch_bytes)
                        self._set_error(tp, err)
                        tp_state.consumed_to(tp_state.position + 1)
                        needs_wakeup = True
                elif error_type in (Errors.NotLeaderForPartitionError, Errors.UnknownTopicOrPartitionError):
                    self._client.force_metadata_update()
                elif error_type is Errors.OffsetOutOfRangeError:
                    if self._default_reset_strategy != OffsetResetStrategy.NONE:
                        tp_state.await_reset(self._default_reset_strategy)
                    else:
                        err = Errors.OffsetOutOfRangeError({tp: fetch_offset})
                        self._set_error(tp, err)
                        needs_wakeup = True
                    log.info('Fetch offset %s is out of range for partition %s, resetting offset', fetch_offset, tp)
                elif error_type is Errors.TopicAuthorizationFailedError:
                    log.warning('Not authorized to read from topic %s.', tp.topic)
                    err = Errors.TopicAuthorizationFailedError(tp.topic)
                    self._set_error(tp, err)
                    needs_wakeup = True
                else:
                    log.warning('Unexpected error while fetching data: %s', error_type.__name__)
        return needs_wakeup

    def _set_error(self, tp, error):
        assert tp not in self._records, self._records[tp]
        self._records[tp] = FetchError(error=error, backoff=self._prefetch_backoff)

    async def _update_fetch_positions(self, assignment, node_id, tps):
        """ This task will be called if there is no valid position for
        partition. It may be right after assignment, on seek_to_* calls of
        Consumer or if current position went out of range.
        """
        log.debug('Updating fetch positions for partitions %s', tps)
        needs_wakeup = False
        for tp in tps:
            tp_state = assignment.state_value(tp)
            if tp_state.has_valid_position or tp_state.awaiting_reset:
                continue
            try:
                committed = await tp_state.fetch_committed()
            except asyncio.CancelledError:
                return needs_wakeup
            assert committed is not None
            if tp_state.has_valid_position or tp_state.awaiting_reset:
                continue
            if committed.offset == UNKNOWN_OFFSET:
                if self._default_reset_strategy != OffsetResetStrategy.NONE:
                    tp_state.await_reset(self._default_reset_strategy)
                else:
                    err = Errors.NoOffsetForPartitionError(tp)
                    self._set_error(tp, err)
                    needs_wakeup = True
                log.debug('No committed offset found for %s', tp)
            else:
                log.debug('Resetting offset for partition %s to the committed offset %s', tp, committed)
                tp_state.reset_to(committed.offset)
        topic_data = collections.defaultdict(list)
        needs_reset = []
        for tp in tps:
            tp_state = assignment.state_value(tp)
            if not tp_state.awaiting_reset:
                continue
            needs_reset.append(tp)
            strategy = tp_state.reset_strategy
            assert strategy is not None
            log.debug('Resetting offset for partition %s using %s strategy.', tp, OffsetResetStrategy.to_str(strategy))
            topic_data[tp.topic].append((tp.partition, strategy))
        if not topic_data:
            return needs_wakeup
        try:
            try:
                offsets = await self._proc_offset_request(node_id, topic_data)
            except Errors.KafkaError as err:
                log.error('Failed fetch offsets from %s: %s', node_id, err)
                await asyncio.sleep(self._retry_backoff)
                return needs_wakeup
        except asyncio.CancelledError:
            return needs_wakeup
        for tp in needs_reset:
            offset = offsets[tp][0]
            tp_state = assignment.state_value(tp)
            if tp_state.awaiting_reset:
                tp_state.reset_to(offset)
        return needs_wakeup

    async def _retrieve_offsets(self, timestamps, timeout_ms=None):
        """ Fetch offset for each partition passed in ``timestamps`` map.

        Blocks until offsets are obtained, a non-retriable exception is raised
        or ``timeout_ms`` passed.

        Arguments:
            timestamps: {TopicPartition: int} dict with timestamps to fetch
                offsets by. -1 for the latest available, -2 for the earliest
                available. Otherwise timestamp is treated as epoch
                milliseconds.

        Returns:
            {TopicPartition: (int, int)}: Mapping of partition to
                retrieved offset and timestamp. If offset does not exist for
                the provided timestamp, that partition will be missing from
                this mapping.
        """
        if not timestamps:
            return {}
        timeout = None if timeout_ms is None else timeout_ms / 1000
        try:
            async with async_timeout.timeout(timeout):
                while True:
                    try:
                        offsets = await self._proc_offset_requests(timestamps)
                    except Errors.KafkaError as error:
                        if not error.retriable:
                            raise error
                        if error.invalid_metadata:
                            self._client.force_metadata_update()
                        await asyncio.sleep(self._retry_backoff)
                    else:
                        return offsets
        except asyncio.TimeoutError:
            raise KafkaTimeoutError('Failed to get offsets by times in %s ms' % timeout_ms)

    async def _proc_offset_requests(self, timestamps):
        """ Fetch offsets for each partition in timestamps dict. This may send
        request to multiple nodes, based on who is Leader for partition.

        Arguments:
            timestamps (dict): {TopicPartition: int} mapping of fetching
                timestamps.

        Returns:
            Future: resolves to a mapping of retrieved offsets
        """
        await self._client._maybe_wait_metadata()
        timestamps_by_node = collections.defaultdict(lambda: collections.defaultdict(list))
        for partition, timestamp in timestamps.items():
            node_id = self._client.cluster.leader_for_partition(partition)
            if node_id is None:
                self._client.add_topic(partition.topic)
                log.debug('Partition %s is unknown for fetching offset, wait for metadata refresh', partition)
                raise Errors.StaleMetadata(partition)
            elif node_id == -1:
                log.debug('Leader for partition %s unavailable for fetching offset, wait for metadata refresh', partition)
                raise Errors.LeaderNotAvailableError(partition)
            else:
                timestamps_by_node[node_id][partition.topic].append((partition.partition, timestamp))
        futs = []
        for node_id, topic_data in timestamps_by_node.items():
            futs.append(self._proc_offset_request(node_id, topic_data))
        offsets = {}
        res = await asyncio.gather(*futs)
        for partial_offsets in res:
            offsets.update(partial_offsets)
        return offsets

    async def _proc_offset_request(self, node_id, topic_data):
        if self._client.api_version < (0, 10, 1):
            version = 0
            for topic, part_data in topic_data.items():
                topic_data[topic] = [(part, ts, 1) for part, ts in part_data]
        else:
            version = 1
        request = OffsetRequest[version](-1, list(topic_data.items()))
        response = await self._client.send(node_id, request)
        res_offsets = {}
        for topic, part_data in response.topics:
            for part, error_code, *partition_info in part_data:
                partition = TopicPartition(topic, part)
                error_type = Errors.for_code(error_code)
                if error_type is Errors.NoError:
                    if response.API_VERSION == 0:
                        offsets = partition_info[0]
                        assert len(offsets) <= 1, 'Expected OffsetResponse with one offset'
                        if offsets:
                            offset = offsets[0]
                            log.debug('Handling v0 ListOffsetResponse response for %s. Fetched offset %s', partition, offset)
                            res_offsets[partition] = (offset, None)
                        else:
                            res_offsets[partition] = (UNKNOWN_OFFSET, None)
                    else:
                        timestamp, offset = partition_info
                        log.debug('Handling ListOffsetResponse response for %s. Fetched offset %s, timestamp %s', partition, offset, timestamp)
                        res_offsets[partition] = (offset, timestamp)
                elif error_type is Errors.UnsupportedForMessageFormatError:
                    log.debug('Cannot search by timestamp for partition %s because the message format version is before 0.10.0', partition)
                elif error_type is Errors.NotLeaderForPartitionError:
                    log.debug('Attempt to fetch offsets for partition %s failed due to obsolete leadership information, retrying.', partition)
                    raise error_type(partition)
                elif error_type is Errors.UnknownTopicOrPartitionError:
                    log.warning('Received unknown topic or partition error in ListOffset request for partition %s. The topic/partition may not exist or the user may not have Describe access to it.', partition)
                    raise error_type(partition)
                else:
                    log.warning('Attempt to fetch offsets for partition %s failed due to: %s', partition, error_type)
                    raise error_type(partition)
        return res_offsets

    async def next_record(self, partitions):
        """ Return one fetched records

        This method will contain a little overhead as we will do more work this
        way:
            * Notify prefetch routine per every consumed partition
            * Assure message marked for autocommit

        """
        while True:
            if self._closed:
                raise ConsumerStoppedError()
            if self._subscriptions.reassignment_in_progress:
                await self._subscriptions.wait_for_assignment()
            for tp in list(self._records.keys()):
                if partitions and tp not in partitions:
                    if not self._subscriptions.is_assigned(tp):
                        del self._records[tp]
                    continue
                res_or_error = self._records[tp]
                if type(res_or_error) == FetchResult:
                    message = res_or_error.getone()
                    if message is None:
                        del self._records[tp]
                        self._notify(self._wait_consume_future)
                    else:
                        return message
                else:
                    del self._records[tp]
                    self._notify(self._wait_consume_future)
                    res_or_error.check_raise()
            waiter = self._create_fetch_waiter()
            await waiter

    async def fetched_records(self, partitions, timeout=0, max_records=None):
        """ Returns previously fetched records and updates consumed offsets.
        """
        while True:
            if self._subscriptions.reassignment_in_progress:
                await self._subscriptions.wait_for_assignment()
            start_time = time.monotonic()
            drained = {}
            for tp in list(self._records.keys()):
                if partitions and tp not in partitions:
                    if not self._subscriptions.is_assigned(tp):
                        del self._records[tp]
                    continue
                res_or_error = self._records[tp]
                if type(res_or_error) == FetchResult:
                    records = res_or_error.getall(max_records)
                    if not res_or_error.has_more():
                        del self._records[tp]
                        self._notify(self._wait_consume_future)
                    if not records:
                        continue
                    drained[tp] = records
                    if max_records is not None:
                        max_records -= len(drained[tp])
                        assert max_records >= 0
                        if max_records == 0:
                            break
                elif drained:
                    return drained
                else:
                    del self._records[tp]
                    self._notify(self._wait_consume_future)
                    res_or_error.check_raise()
            if drained or not timeout:
                return drained
            waiter = self._create_fetch_waiter()
            done, pending = await asyncio.wait([waiter], timeout=timeout)
            if not done or self._closed:
                if pending:
                    fut = pending.pop()
                    fut.cancel()
                return {}
            if waiter.done():
                waiter.result()
            timeout = timeout - (time.monotonic() - start_time)
            timeout = max(0, timeout)

    async def get_offsets_by_times(self, timestamps, timeout_ms):
        offsets = await self._retrieve_offsets(timestamps, timeout_ms)
        for tp in timestamps:
            if tp not in offsets:
                offsets[tp] = None
            else:
                offset, timestamp = offsets[tp]
                if offset == UNKNOWN_OFFSET:
                    offsets[tp] = None
                else:
                    offsets[tp] = OffsetAndTimestamp(offset, timestamp)
        return offsets

    async def beginning_offsets(self, partitions, timeout_ms):
        timestamps = {tp: OffsetResetStrategy.EARLIEST for tp in partitions}
        offsets = await self._retrieve_offsets(timestamps, timeout_ms)
        return {tp: offset for tp, (offset, ts) in offsets.items()}

    async def end_offsets(self, partitions, timeout_ms):
        timestamps = {tp: OffsetResetStrategy.LATEST for tp in partitions}
        offsets = await self._retrieve_offsets(timestamps, timeout_ms)
        return {tp: offset for tp, (offset, ts) in offsets.items()}

    def request_offset_reset(self, tps, strategy):
        """ Force a position reset. Called from Consumer of `seek_to_*` API's.
        """
        assignment = self._subscriptions.subscription.assignment
        assert assignment is not None
        waiters = []
        for tp in tps:
            tp_state = assignment.state_value(tp)
            tp_state.await_reset(strategy)
            waiters.append(tp_state.wait_for_position())
            if tp in self._records:
                del self._records[tp]
        self._notify(self._wait_consume_future)
        return asyncio.gather(*waiters)

    def seek_to(self, tp, offset):
        """ Force a position change to specific offset. Called from
        `Consumer.seek()` API.
        """
        self._subscriptions.seek(tp, offset)
        if tp in self._records:
            del self._records[tp]
        self._notify(self._wait_consume_future)