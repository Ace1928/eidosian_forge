import datetime
import logging
import secrets
import sys
import threading
from typing import TYPE_CHECKING, Optional, cast
def _log_message_assert(msg: 'MessageType') -> None:
    _log_message(msg, 'assert')