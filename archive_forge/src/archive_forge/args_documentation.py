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

    Helper to execute code in a given interpreter, e.g. to implement the behavior of python3 [-i] file.py

    args should be a [faked] sys.argv.
    