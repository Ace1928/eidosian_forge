import asyncio
import enum
import io
import json
import mimetypes
import os
import warnings
from abc import ABC, abstractmethod
from itertools import chain
from typing import (
from multidict import CIMultiDict
from . import hdrs
from .abc import AbstractStreamWriter
from .helpers import (
from .streams import StreamReader
from .typedefs import JSONEncoder, _CIMultiDict
class payload_type:

    def __init__(self, type: Any, *, order: Order=Order.normal) -> None:
        self.type = type
        self.order = order

    def __call__(self, factory: Type['Payload']) -> Type['Payload']:
        register_payload(factory, self.type, order=self.order)
        return factory