from __future__ import annotations
import os
import argparse
import multiprocessing
import subprocess
from pathlib import Path
import typing as T
from ..mesonlib import Popen_safe, split_args
def write_depfile(self) -> None:
    with open(self.depfile, 'w', encoding='utf-8') as f:
        f.write(f'{self.stampfile}: \\\n')
        for dirpath, dirnames, filenames in os.walk(self.src_dir):
            dirnames[:] = [d for d in dirnames if not d.startswith('.')]
            for fname in filenames:
                if fname.startswith('.'):
                    continue
                path = Path(dirpath, fname)
                f.write('  {} \\\n'.format(path.as_posix().replace(' ', '\\ ')))