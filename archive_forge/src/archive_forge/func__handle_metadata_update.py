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
def _handle_metadata_update(self, cluster):
    subscription = self._subscription
    if subscription.subscribed_pattern:
        topics = []
        for topic in cluster.topics(self._exclude_internal_topics):
            if subscription.subscribed_pattern.match(topic):
                topics.append(topic)
        if subscription.subscription is None or set(topics) != subscription.subscription.topics:
            subscription.subscribe_from_pattern(topics)
    if subscription.partitions_auto_assigned() and self._group_subscription is not None:
        metadata_snapshot = self._get_metadata_snapshot()
        if self._metadata_snapshot != metadata_snapshot:
            log.info('Metadata for topic has changed from %s to %s. ', self._metadata_snapshot, metadata_snapshot)
            self._metadata_snapshot = metadata_snapshot
            self._on_metadata_change()