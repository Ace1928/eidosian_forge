from __future__ import annotations
import sys
from collections import UserDict
from typing import TYPE_CHECKING, Union
from urllib.parse import quote as _quote, unquote as _unquote
from twisted.internet import defer, protocol
from twisted.protocols import basic, policies
from twisted.python import log
class PostfixTCPMapDictServerFactory(_PostfixTCPMapDict, protocol.ServerFactory):
    """
    An in-memory dictionary factory for PostfixTCPMapServer.
    """
    protocol = PostfixTCPMapServer