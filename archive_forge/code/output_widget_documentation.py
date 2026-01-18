from typing import Any, Dict, List, Optional
from jupyter_client.client import KernelClient
from nbformat.v4 import output_from_msg
from .jsonutil import json_clean
Handle a message.