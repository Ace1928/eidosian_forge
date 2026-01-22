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
def _finalize_packet(self, packet_char, packet_width):
    self._chars[packet_char] = Page(text=self.text, boxes=self.boxes, width=packet_width, height=None, descent=None)
    self.state = _dvistate.outer