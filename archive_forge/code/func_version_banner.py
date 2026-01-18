import argparse
from typing import Tuple, List, Optional, NoReturn, Callable
import code
import curtsies
import cwcwidth
import greenlet
import importlib.util
import logging
import os
import pygments
import requests
import sys
import xdg
from pathlib import Path
from . import __version__, __copyright__
from .config import default_config_path, Config
from .translations import _
def version_banner(base: str='bpython') -> str:
    return _('{} version {} on top of Python {} {}').format(base, __version__, sys.version.split()[0], sys.executable)