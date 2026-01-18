from __future__ import annotations
from argparse import ArgumentParser
from argparse import Namespace
import contextlib
import difflib
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
from typing import Any
from typing import Dict
from typing import Iterator
from typing import Optional
from typing import Union
from . import compat
def write_output_file_from_text(self, text: str, destination_path: Union[str, Path]) -> None:
    if self.args.check:
        self._run_diff(destination_path, source=text)
    elif self.args.stdout:
        print(text)
    else:
        self.write_status(f'Writing {destination_path}...')
        Path(destination_path).write_text(text, encoding='utf-8', newline='\n')
        self.write_status('done\n')