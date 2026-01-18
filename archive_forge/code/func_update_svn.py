from __future__ import annotations
from dataclasses import dataclass, InitVar
import os, subprocess
import argparse
import asyncio
import threading
import copy
import shutil
from concurrent.futures.thread import ThreadPoolExecutor
from pathlib import Path
import typing as T
import tarfile
import zipfile
from . import mlog
from .ast import IntrospectionInterpreter
from .mesonlib import quiet_git, GitException, Popen_safe, MesonException, windows_proof_rmtree
from .wrap.wrap import (Resolver, WrapException, ALL_TYPES,
def update_svn(self) -> bool:
    revno = self.wrap.get('revision')
    _, out, _ = Popen_safe(['svn', 'info', '--show-item', 'revision', self.repo_dir])
    current_revno = out
    if current_revno == revno:
        return True
    if revno.lower() == 'head':
        subprocess.call(['svn', 'update'], cwd=self.repo_dir)
    else:
        subprocess.check_call(['svn', 'update', '-r', revno], cwd=self.repo_dir)
    return True