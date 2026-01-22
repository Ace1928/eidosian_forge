from enum import Enum
from collections import namedtuple, defaultdict, deque
from aiokafka.structs import TopicPartition
from aiokafka.util import create_future
def aborting_transaction(self):
    self._transition_to(TransactionState.ABORTING_TRANSACTION)
    if self._transaction_waiter.done():
        self._transaction_waiter = create_future()
    self.notify_task_waiter()