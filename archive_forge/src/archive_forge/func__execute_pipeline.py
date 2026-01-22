import copy
import re
import threading
import time
import warnings
from itertools import chain
from typing import Any, Callable, Dict, List, Optional, Type, Union
from redis._parsers.encoders import Encoder
from redis._parsers.helpers import (
from redis.commands import (
from redis.connection import (
from redis.credentials import CredentialProvider
from redis.exceptions import (
from redis.lock import Lock
from redis.retry import Retry
from redis.utils import (
def _execute_pipeline(self, connection, commands, raise_on_error):
    all_cmds = connection.pack_commands([args for args, _ in commands])
    connection.send_packed_command(all_cmds)
    response = []
    for args, options in commands:
        try:
            response.append(self.parse_response(connection, args[0], **options))
        except ResponseError as e:
            response.append(e)
    if raise_on_error:
        self.raise_first_error(commands, response)
    return response