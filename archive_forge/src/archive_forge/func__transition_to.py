from enum import Enum
from collections import namedtuple, defaultdict, deque
from aiokafka.structs import TopicPartition
from aiokafka.util import create_future
def _transition_to(self, target):
    assert TransactionState.is_transition_valid(self.state, target), f'Invalid state transition {self.state} -> {target}'
    self.state = target