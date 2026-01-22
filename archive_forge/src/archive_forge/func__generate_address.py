import secrets
import string
import threading
import time
from typing import TYPE_CHECKING, Callable, Dict, List, Optional, Tuple
from wandb.errors import Error
from wandb.proto import wandb_internal_pb2 as pb
def _generate_address(length: int=12) -> str:
    address = ''.join((secrets.choice(string.ascii_lowercase + string.digits) for i in range(length)))
    return address