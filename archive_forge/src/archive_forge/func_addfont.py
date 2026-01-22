from base64 import b64encode
from collections import namedtuple
import copy
import dataclasses
from functools import lru_cache
from io import BytesIO
import json
import logging
from numbers import Number
import os
from pathlib import Path
import re
import subprocess
import sys
import threading
from typing import Union
import matplotlib as mpl
from matplotlib import _api, _afm, cbook, ft2font
from matplotlib._fontconfig_pattern import (
from matplotlib.rcsetup import _validators
def addfont(self, path):
    """
        Cache the properties of the font at *path* to make it available to the
        `FontManager`.  The type of font is inferred from the path suffix.

        Parameters
        ----------
        path : str or path-like

        Notes
        -----
        This method is useful for adding a custom font without installing it in
        your operating system. See the `FontManager` singleton instance for
        usage and caveats about this function.
        """
    path = os.fsdecode(path)
    if Path(path).suffix.lower() == '.afm':
        with open(path, 'rb') as fh:
            font = _afm.AFM(fh)
        prop = afmFontProperty(path, font)
        self.afmlist.append(prop)
    else:
        font = ft2font.FT2Font(path)
        prop = ttfFontProperty(font)
        self.ttflist.append(prop)
    self._findfont_cached.cache_clear()