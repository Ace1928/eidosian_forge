import queue
from typing import TYPE_CHECKING, Optional
from ..lib import tracelog
from ..lib.mailbox import Mailbox
from .router import MessageRouter
Router - handle message router (queue).

Router to manage responses from a queue.

