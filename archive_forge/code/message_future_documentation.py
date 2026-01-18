import threading
from abc import abstractmethod
from typing import Optional
from wandb.proto import wandb_internal_pb2 as pb
MessageFuture - represents a message result of an asynchronous operation.

Base class MessageFuture for MessageFutureObject and MessageFuturePoll

