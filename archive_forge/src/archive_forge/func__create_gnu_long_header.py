from builtins import open as bltn_open
import sys
import os
import io
import shutil
import stat
import time
import struct
import copy
import re
import warnings
@classmethod
def _create_gnu_long_header(cls, name, type, encoding, errors):
    """Return a GNUTYPE_LONGNAME or GNUTYPE_LONGLINK sequence
           for name.
        """
    name = name.encode(encoding, errors) + NUL
    info = {}
    info['name'] = '././@LongLink'
    info['type'] = type
    info['size'] = len(name)
    info['magic'] = GNU_MAGIC
    return cls._create_header(info, USTAR_FORMAT, encoding, errors) + cls._create_payload(name)