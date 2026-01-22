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
def _fail_all(self, task):
    """ Called when sender fails. Will fail all pending batches, as they
        will never be delivered as well as fail transaction
        """
    if task.cancelled():
        return
    task_exception = task.exception()
    if task_exception is not None:
        self._message_accumulator.fail_all(task_exception)
        if self._txn_manager is not None:
            self._txn_manager.fatal_error(task_exception)