import logging
from multiprocessing.process import BaseProcess
from typing import TYPE_CHECKING, Optional
from wandb.proto import wandb_internal_pb2 as pb
from ..lib.mailbox import Mailbox
from .interface_queue import InterfaceQueue
from .router_relay import MessageRelayRouter
InterfaceRelay - Derived from InterfaceQueue using RelayRouter to preserve uuid req/resp.

See interface.py for how interface classes relate to each other.

