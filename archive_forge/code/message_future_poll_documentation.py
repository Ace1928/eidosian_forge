import time
from typing import Any, Optional
from wandb.proto import wandb_internal_pb2 as pb
from .message_future import MessageFuture
MessageFuturePoll - Derived from MessageFuture but implementing polling loop.

MessageFuture represents a message result of an asynchronous operation.

MessageFuturePoll implements a polling loop to periodically query for a
completed async operation.

