import os
import struct
from codecs import getincrementaldecoder, IncrementalDecoder
from enum import IntEnum
from typing import Generator, List, NamedTuple, Optional, Tuple, TYPE_CHECKING, Union
class ParseFailed(Exception):

    def __init__(self, msg: str, code: CloseReason=CloseReason.PROTOCOL_ERROR) -> None:
        super().__init__(msg)
        self.code = code