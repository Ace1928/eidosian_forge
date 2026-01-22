import json
import logging
import os
import subprocess
from typing import Any, Dict, List, Optional, Tuple, Union
import requests
from dockerpycreds.utils import find_executable  # type: ignore
from wandb.docker import auth, www_authenticate
from wandb.errors import Error
def _post_process_stream(stream: Optional[bytes]) -> str:
    if stream is None:
        return ''
    decoded_stream = stream.decode()
    if len(decoded_stream) != 0 and decoded_stream[-1] == '\n':
        decoded_stream = decoded_stream[:-1]
    return decoded_stream