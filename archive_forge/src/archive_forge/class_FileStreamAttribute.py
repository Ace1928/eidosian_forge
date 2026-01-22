from __future__ import annotations
from collections.abc import Callable, Mapping
from io import SEEK_SET, UnsupportedOperation
from os import PathLike
from pathlib import Path
from typing import Any, BinaryIO, cast
from .. import (
from ..abc import ByteReceiveStream, ByteSendStream
class FileStreamAttribute(TypedAttributeSet):
    file: BinaryIO = typed_attribute()
    path: Path = typed_attribute()
    fileno: int = typed_attribute()