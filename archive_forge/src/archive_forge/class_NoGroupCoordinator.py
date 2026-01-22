import asyncio
import collections
import logging
import copy
import time
import aiokafka.errors as Errors
from aiokafka.client import ConnectionGroup, CoordinationType
from aiokafka.coordinator.assignors.roundrobin import RoundRobinPartitionAssignor
from aiokafka.coordinator.protocol import ConsumerProtocol
from aiokafka.protocol.api import Response
from aiokafka.protocol.commit import (
from aiokafka.protocol.group import (
from aiokafka.structs import OffsetAndMetadata, TopicPartition
from aiokafka.util import create_future, create_task
class NoGroupCoordinator(BaseCoordinator):
    """
    When `group_id` consumer option is not used we don't have the functionality
    provided by Coordinator node in Kafka cluster, like committing offsets (
    Kafka based offset storage) or automatic partition assignment between
    consumers. But `GroupCoordinator` class has some other responsibilities,
    that this class takes care of to avoid code duplication, like:

        * Static topic partition assignment when we subscribed to topic.
          Partition changes will be noticed by metadata update and assigned.
        * Pattern topic subscription. New topics will be noticed by metadata
          update and added to subscription.
    """

    def __init__(self, *args, **kw):
        super().__init__(*args, **kw)
        self._reset_committed_task = create_task(self._reset_committed_routine())

    def _on_metadata_change(self):
        self.assign_all_partitions()

    def assign_all_partitions(self, check_unknown=False):
        """ Assign all partitions from subscribed topics to this consumer.
            If `check_unknown` we will raise UnknownTopicOrPartitionError if
            subscribed topic is not found in metadata response.
        """
        partitions = []
        for topic in self._subscription.subscription.topics:
            p_ids = self._cluster.partitions_for_topic(topic)
            if not p_ids:
                if check_unknown:
                    raise Errors.UnknownTopicOrPartitionError()
                else:
                    continue
            for p_id in p_ids:
                partitions.append(TopicPartition(topic, p_id))
        assignment = self._subscription.subscription.assignment
        if assignment is None or set(partitions) != assignment.tps:
            self._subscription.assign_from_subscribed(partitions)

    async def _reset_committed_routine(self):
        """ Group coordinator will reset committed points to UNKNOWN_OFFSET
        if no commit is found for group. In the NoGroup mode we need to force
        it after each assignment
        """
        event_waiter = None
        try:
            while True:
                if self._subscription.subscription is None:
                    await self._subscription.wait_for_subscription()
                    continue
                assignment = self._subscription.subscription.assignment
                if assignment is None:
                    await self._subscription.wait_for_assignment()
                    continue
                commit_refresh_needed = assignment.commit_refresh_needed
                commit_refresh_needed.clear()
                for tp in assignment.requesting_committed():
                    tp_state = assignment.state_value(tp)
                    tp_state.update_committed(OffsetAndMetadata(UNKNOWN_OFFSET, ''))
                event_waiter = create_task(commit_refresh_needed.wait())
                await asyncio.wait([assignment.unassign_future, event_waiter], return_when=asyncio.FIRST_COMPLETED)
                if not event_waiter.done():
                    event_waiter.cancel()
                    event_waiter = None
        except asyncio.CancelledError:
            pass
        if event_waiter is not None and (not event_waiter.done()):
            event_waiter.cancel()
            event_waiter = None

    @property
    def _group_subscription(self):
        return self._subscription.subscription.topics

    async def close(self):
        self._reset_committed_task.cancel()
        await self._reset_committed_task
        self._reset_committed_task = None

    def check_errors(self):
        if self._reset_committed_task.done():
            self._reset_committed_task.result()