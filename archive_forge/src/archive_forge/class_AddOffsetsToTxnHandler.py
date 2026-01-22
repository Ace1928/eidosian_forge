import asyncio
import collections
import logging
import time
import aiokafka.errors as Errors
from aiokafka.client import ConnectionGroup, CoordinationType
from aiokafka.errors import (
from aiokafka.protocol.produce import ProduceRequest
from aiokafka.protocol.transaction import (
from aiokafka.structs import TopicPartition
from aiokafka.util import create_task
class AddOffsetsToTxnHandler(BaseHandler):
    group = ConnectionGroup.COORDINATION

    def __init__(self, sender, group_id):
        super().__init__(sender)
        self._group_id = group_id

    def create_request(self):
        txn_manager = self._sender._txn_manager
        req = AddOffsetsToTxnRequest[0](transactional_id=txn_manager.transactional_id, producer_id=txn_manager.producer_id, producer_epoch=txn_manager.producer_epoch, group_id=self._group_id)
        return req

    def handle_response(self, resp):
        txn_manager = self._sender._txn_manager
        group_id = self._group_id
        error_type = Errors.for_code(resp.error_code)
        if error_type is Errors.NoError:
            log.debug('Successfully added consumer group %s to transaction', group_id)
            txn_manager.consumer_group_added(group_id)
            return
        elif error_type is CoordinatorNotAvailableError or error_type is NotCoordinatorError:
            self._sender._coordinator_dead(CoordinationType.TRANSACTION)
        elif error_type is CoordinatorLoadInProgressError or error_type is ConcurrentTransactions:
            pass
        elif error_type is InvalidProducerEpoch:
            raise ProducerFenced()
        elif error_type is InvalidTxnState:
            raise error_type()
        elif error_type is TransactionalIdAuthorizationFailed:
            raise error_type(txn_manager.transactional_id)
        elif error_type is GroupAuthorizationFailedError:
            txn_manager.error_transaction(error_type(self._group_id))
            return
        else:
            log.error('Could not add consumer group due to unexpected error: %s', error_type)
            raise error_type()
        return self._default_backoff