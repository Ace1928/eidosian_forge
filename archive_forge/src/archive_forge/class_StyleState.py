import argparse
import collections
import functools
import glob
import inspect
import itertools
import os
import re
import subprocess
import sys
import threading
import unicodedata
from enum import (
from typing import (
from . import (
from .argparse_custom import (
class StyleState:
    """Keeps track of what text styles are enabled"""

    def __init__(self) -> None:
        self.style_dict: Dict[int, str] = dict()
        self.reset_all: Optional[int] = None
        self.fg: Optional[int] = None
        self.bg: Optional[int] = None
        self.intensity: Optional[int] = None
        self.italic: Optional[int] = None
        self.overline: Optional[int] = None
        self.strikethrough: Optional[int] = None
        self.underline: Optional[int] = None