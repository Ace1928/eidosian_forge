import struct
import time
from binascii import crc32
import aiokafka.codec as codecs
from aiokafka.codec import (
from aiokafka.errors import CorruptRecordException, UnsupportedCodecError
from aiokafka.util import NO_EXTENSIONS
 Actual size of message to add
        