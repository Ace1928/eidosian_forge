from __future__ import annotations
import enum
import os
import io
import sys
import time
import platform
import shlex
import subprocess
import shutil
import typing as T
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
class AnsiText:

    def __init__(self, *args: 'SizedStringProtocol'):
        self.args = args

    def __len__(self) -> int:
        return sum((len(x) for x in self.args))

    def __str__(self) -> str:
        return ''.join((str(x) for x in self.args))