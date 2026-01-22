import base64
import imghdr
from collections import OrderedDict
from os import path
from typing import IO, BinaryIO, NamedTuple, Optional, Tuple
import imagesize
class DataURI(NamedTuple):
    mimetype: str
    charset: str
    data: bytes