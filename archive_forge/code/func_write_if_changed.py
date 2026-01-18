from __future__ import annotations
import typing as T
import os, sys
from .. import mesonlib
from .. import mlog
from ..mesonlib import Popen_safe
import argparse
def write_if_changed(text: str, outfilename: str) -> None:
    try:
        with open(outfilename, encoding='utf-8') as f:
            oldtext = f.read()
        if text == oldtext:
            return
    except FileNotFoundError:
        pass
    with open(outfilename, 'w', encoding='utf-8') as f:
        f.write(text)