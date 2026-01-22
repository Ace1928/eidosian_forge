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
class PythonRespSerializer:

    def __init__(self, buffer_cutoff, encode) -> None:
        self._buffer_cutoff = buffer_cutoff
        self.encode = encode

    def pack(self, *args):
        """Pack a series of arguments into the Redis protocol"""
        output = []
        if isinstance(args[0], str):
            args = tuple(args[0].encode().split()) + args[1:]
        elif b' ' in args[0]:
            args = tuple(args[0].split()) + args[1:]
        buff = SYM_EMPTY.join((SYM_STAR, str(len(args)).encode(), SYM_CRLF))
        buffer_cutoff = self._buffer_cutoff
        for arg in map(self.encode, args):
            arg_length = len(arg)
            if len(buff) > buffer_cutoff or arg_length > buffer_cutoff or isinstance(arg, memoryview):
                buff = SYM_EMPTY.join((buff, SYM_DOLLAR, str(arg_length).encode(), SYM_CRLF))
                output.append(buff)
                output.append(arg)
                buff = SYM_CRLF
            else:
                buff = SYM_EMPTY.join((buff, SYM_DOLLAR, str(arg_length).encode(), SYM_CRLF, arg, SYM_CRLF))
        output.append(buff)
        return output