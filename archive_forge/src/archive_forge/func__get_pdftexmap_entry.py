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
def _get_pdftexmap_entry(self):
    return PsfontsMap(find_tex_file('pdftex.map'))[self.font.texname]