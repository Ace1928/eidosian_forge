import copy
import os
import socket
import ssl
import sys
import threading
import weakref
from abc import abstractmethod
from itertools import chain
from queue import Empty, Full, LifoQueue
from time import time
from typing import Any, Callable, List, Optional, Type, Union
from urllib.parse import parse_qs, unquote, urlparse
from ._parsers import Encoder, _HiredisParser, _RESP2Parser, _RESP3Parser
from .backoff import NoBackoff
from .credentials import CredentialProvider, UsernamePasswordCredentialProvider
from .exceptions import (
from .retry import Retry
from .utils import (
def _construct_command_packer(self, packer):
    if packer is not None:
        return packer
    elif HIREDIS_PACK_AVAILABLE:
        return HiredisRespSerializer()
    else:
        return PythonRespSerializer(self._buffer_cutoff, self.encoder.encode)