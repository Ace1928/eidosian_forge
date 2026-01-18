from __future__ import annotations
import os
import argparse
import multiprocessing
import subprocess
from pathlib import Path
import typing as T
from ..mesonlib import Popen_safe, split_args
def write_stampfile(self) -> None:
    with open(self.stampfile, 'w', encoding='utf-8'):
        pass