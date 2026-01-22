import os
from typing import Optional, Union
from warnings import warn
from zope.interface import implementer
import attr
from typing_extensions import Literal
from twisted.internet.interfaces import IAddress
from twisted.python.filepath import _asFilesystemBytes, _coerceToFilesystemEncoding
from twisted.python.runtime import platform
@implementer(IAddress)
class _ProcessAddress:
    """
    An L{interfaces.IAddress} provider for process transports.
    """