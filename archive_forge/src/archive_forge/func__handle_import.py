import ast
import json
import os
import sys
import tokenize
from argparse import ArgumentParser, RawTextHelpFormatter
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Union
def _handle_import(self, mod: str):
    if mod.startswith('PySide6.'):
        self._qt_modules.add(mod[8:])