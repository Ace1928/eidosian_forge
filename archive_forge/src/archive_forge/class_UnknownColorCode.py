import os
import sys
import locale
from configparser import ConfigParser
from itertools import chain
from pathlib import Path
from typing import MutableMapping, Mapping, Any, Dict
from xdg import BaseDirectory
from .autocomplete import AutocompleteModes
class UnknownColorCode(Exception):

    def __init__(self, key: str, color: str) -> None:
        self.key = key
        self.color = color