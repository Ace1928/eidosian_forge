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
class CoordinatorGroupRebalance:
    """ An adapter, that encapsulates rebalance logic and will have a copy of
        assigned topics, so we can detect assignment changes. This includes
        subscription pattern changes.

        On how to handle cases read in https://cwiki.apache.org/confluence/            display/KAFKA/Kafka+Client-side+Assignment+Proposal
    """

    def __init__(self, coordinator, group_id, coordinator_id, subscription, assignors, session_timeout_ms, retry_backoff_ms):
        self._coordinator = coordinator
        self.group_id = group_id
        self.coordinator_id = coordinator_id
        self._subscription = subscription
        self._assignors = assignors
        self._session_timeout_ms = session_timeout_ms
        self._retry_backoff_ms = retry_backoff_ms
        self._api_version = self._coordinator._client.api_version
        self._rebalance_timeout_ms = self._coordinator._rebalance_timeout_ms

    async def perform_group_join(self):
        """Join the group and return the assignment for the next generation.

        This function handles both JoinGroup and SyncGroup, delegating to
        _perform_assignment() if elected as leader by the coordinator node.

        Returns encoded-bytes assignment returned from the group leader
        """
        log.info('(Re-)joining group %s', self.group_id)
        topics = self._subscription.topics
        metadata_list = []
        for assignor in self._assignors:
            metadata = assignor.metadata(topics)
            if not isinstance(metadata, bytes):
                metadata = metadata.encode()
            group_protocol = (assignor.name, metadata)
            metadata_list.append(group_protocol)
            try_join = True
            while try_join:
                try_join = False
                if self._api_version < (0, 10, 1):
                    request = JoinGroupRequest[0](self.group_id, self._session_timeout_ms, self._coordinator.member_id, ConsumerProtocol.PROTOCOL_TYPE, metadata_list)
                elif self._api_version < (0, 11, 0):
                    request = JoinGroupRequest[1](self.group_id, self._session_timeout_ms, self._rebalance_timeout_ms, self._coordinator.member_id, ConsumerProtocol.PROTOCOL_TYPE, metadata_list)
                elif self._api_version < (2, 3, 0):
                    request = JoinGroupRequest[2](self.group_id, self._session_timeout_ms, self._rebalance_timeout_ms, self._coordinator.member_id, ConsumerProtocol.PROTOCOL_TYPE, metadata_list)
                else:
                    request = JoinGroupRequest[3](self.group_id, self._session_timeout_ms, self._rebalance_timeout_ms, self._coordinator.member_id, self._coordinator._group_instance_id, ConsumerProtocol.PROTOCOL_TYPE, metadata_list)
                log.debug('Sending JoinGroup (%s) to coordinator %s', request, self.coordinator_id)
                try:
                    response = await self._coordinator._send_req(request)
                except Errors.KafkaError:
                    return None
                if not self._subscription.active:
                    return None
                error_type = Errors.for_code(response.error_code)
                if error_type is Errors.MemberIdRequired:
                    self._coordinator.member_id = response.member_id
                    try_join = True
        if error_type is Errors.NoError:
            log.debug('Join group response %s', response)
            self._coordinator.member_id = response.member_id
            self._coordinator.generation = response.generation_id
            protocol = response.group_protocol
            log.info("Joined group '%s' (generation %s) with member_id %s", self.group_id, response.generation_id, response.member_id)
            if response.leader_id == response.member_id:
                log.info('Elected group leader -- performing partition assignments using %s', protocol)
                assignment_bytes = await self._on_join_leader(response)
            else:
                assignment_bytes = await self._on_join_follower()
            if assignment_bytes is None:
                return None
            return (protocol, assignment_bytes)
        elif error_type is Errors.GroupLoadInProgressError:
            log.debug('Attempt to join group %s rejected since coordinator %s is loading the group.', self.group_id, self.coordinator_id)
            await asyncio.sleep(self._retry_backoff_ms / 1000)
        elif error_type is Errors.UnknownMemberIdError:
            self._coordinator.reset_generation()
            log.debug('Attempt to join group %s failed due to unknown member id', self.group_id)
        elif error_type in (Errors.GroupCoordinatorNotAvailableError, Errors.NotCoordinatorForGroupError):
            err = error_type()
            self._coordinator.coordinator_dead()
            log.debug('Attempt to join group %s failed due to obsolete coordinator information: %s', self.group_id, err)
        elif error_type in (Errors.InconsistentGroupProtocolError, Errors.InvalidSessionTimeoutError, Errors.InvalidGroupIdError):
            err = error_type()
            log.error('Attempt to join group failed due to fatal error: %s', err)
            raise err
        elif error_type is Errors.GroupAuthorizationFailedError:
            raise error_type(self.group_id)
        else:
            err = error_type()
            log.error("Unexpected error in join group '%s' response: %s", self.group_id, err)
            raise Errors.KafkaError(repr(err))
        return None

    async def _on_join_follower(self):
        if self._api_version < (2, 3, 0):
            version = 0 if self._api_version < (0, 11, 0) else 1
            request = SyncGroupRequest[version](self.group_id, self._coordinator.generation, self._coordinator.member_id, [])
        else:
            request = SyncGroupRequest[2](self.group_id, self._coordinator.generation, self._coordinator.member_id, self._coordinator._group_instance_id, [])
        log.debug('Sending follower SyncGroup for group %s to coordinator %s: %s', self.group_id, self.coordinator_id, request)
        return await self._send_sync_group_request(request)

    async def _on_join_leader(self, response):
        """
        Perform leader synchronization and send back the assignment
        for the group via SyncGroupRequest

        Arguments:
            response (JoinResponse): broker response to parse

        Returns:
            Future: resolves to member assignment encoded-bytes
        """
        try:
            group_assignment = await self._coordinator._perform_assignment(response)
        except Exception as e:
            raise Errors.KafkaError(repr(e))
        assignment_req = []
        for member_id, assignment in group_assignment.items():
            if not isinstance(assignment, bytes):
                assignment = assignment.encode()
            assignment_req.append((member_id, assignment))
        if self._api_version < (2, 3, 0):
            version = 0 if self._api_version < (0, 11, 0) else 1
            request = SyncGroupRequest[version](self.group_id, self._coordinator.generation, self._coordinator.member_id, assignment_req)
        else:
            request = SyncGroupRequest[2](self.group_id, self._coordinator.generation, self._coordinator.member_id, self._coordinator._group_instance_id, assignment_req)
        log.debug('Sending leader SyncGroup for group %s to coordinator %s: %s', self.group_id, self.coordinator_id, request)
        return await self._send_sync_group_request(request)

    async def _send_sync_group_request(self, request):
        self._coordinator._rejoin_needed_fut = create_future()
        req_generation = self._coordinator.generation
        req_member_id = self._coordinator.member_id
        try:
            response = await self._coordinator._send_req(request)
        except Errors.KafkaError:
            self._coordinator.request_rejoin()
            return None
        error_type = Errors.for_code(response.error_code)
        if error_type is Errors.NoError:
            log.info('Successfully synced group %s with generation %s', self.group_id, self._coordinator.generation)
            self._coordinator.generation = req_generation
            self._coordinator.member_id = req_member_id
            return response.member_assignment
        self._coordinator.request_rejoin()
        if error_type is Errors.RebalanceInProgressError:
            log.debug('SyncGroup for group %s failed due to group rebalance', self.group_id)
        elif error_type in (Errors.UnknownMemberIdError, Errors.IllegalGenerationError):
            err = error_type()
            log.debug('SyncGroup for group %s failed due to %s,', self.group_id, err)
            self._coordinator.reset_generation()
        elif error_type in (Errors.GroupCoordinatorNotAvailableError, Errors.NotCoordinatorForGroupError):
            err = error_type()
            log.debug('SyncGroup for group %s failed due to %s', self.group_id, err)
            self._coordinator.coordinator_dead()
        elif error_type is Errors.GroupAuthorizationFailedError:
            raise error_type(self.group_id)
        else:
            err = error_type()
            log.error('Unexpected error from SyncGroup: %s', err)
            raise Errors.KafkaError(repr(err))
        return None