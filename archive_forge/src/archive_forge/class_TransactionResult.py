from enum import Enum
from collections import namedtuple, defaultdict, deque
from aiokafka.structs import TopicPartition
from aiokafka.util import create_future
class TransactionResult:
    ABORT = 0
    COMMIT = 1