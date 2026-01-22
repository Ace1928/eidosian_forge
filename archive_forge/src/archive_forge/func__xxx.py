from collections import namedtuple
import enum
from functools import lru_cache, partial, wraps
import logging
import os
from pathlib import Path
import re
import struct
import subprocess
import sys
import numpy as np
from matplotlib import _api, cbook
@_dispatch(min=239, max=242, args=('ulen1',))
def _xxx(self, datalen):
    special = self.file.read(datalen)
    _log.debug('Dvi._xxx: encountered special: %s', ''.join([chr(ch) if 32 <= ch < 127 else '<%02x>' % ch for ch in special]))