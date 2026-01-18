import binascii
import os
import mmap
import sys
import time
import errno
from io import BytesIO
from smmap import (
import hashlib
from gitdb.const import (
def to_bin_sha(sha):
    if len(sha) == 20:
        return sha
    return hex_to_bin(sha)