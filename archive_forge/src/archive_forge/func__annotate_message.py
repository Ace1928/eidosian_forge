import datetime
import logging
import secrets
import sys
import threading
from typing import TYPE_CHECKING, Optional, cast
def _annotate_message(msg: 'MessageQueueType') -> None:
    record_id = secrets.token_hex(8)
    msg._info._tracelog_id = record_id