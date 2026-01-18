import logging
from multiprocessing.process import BaseProcess
from typing import TYPE_CHECKING, Optional
from ..lib import tracelog
from ..lib.mailbox import Mailbox
from .interface_shared import InterfaceShared
from .router_queue import MessageQueueRouter
InterfaceQueue - Derived from InterfaceShared using queues to send to internal thread.

See interface.py for how interface classes relate to each other.

