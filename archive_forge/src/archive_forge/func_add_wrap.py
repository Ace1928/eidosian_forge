from __future__ import annotations
from .. import mlog
import contextlib
from dataclasses import dataclass
import urllib.request
import urllib.error
import urllib.parse
import os
import hashlib
import shutil
import tempfile
import stat
import subprocess
import sys
import configparser
import time
import typing as T
import textwrap
import json
from base64 import b64encode
from netrc import netrc
from pathlib import Path, PurePath
from functools import lru_cache
from . import WrapMode
from .. import coredata
from ..mesonlib import quiet_git, GIT, ProgressBar, MesonException, windows_proof_rmtree, Popen_safe
from ..interpreterbase import FeatureNew
from ..interpreterbase import SubProject
from .. import mesonlib
def add_wrap(self, wrap: PackageDefinition) -> None:
    for k in wrap.provided_deps.keys():
        if k in self.provided_deps:
            prev_wrap = self.provided_deps[k]
            m = f'Multiple wrap files provide {k!r} dependency: {wrap.basename} and {prev_wrap.basename}'
            raise WrapException(m)
        self.provided_deps[k] = wrap
    for k in wrap.provided_programs:
        if k in self.provided_programs:
            prev_wrap = self.provided_programs[k]
            m = f'Multiple wrap files provide {k!r} program: {wrap.basename} and {prev_wrap.basename}'
            raise WrapException(m)
        self.provided_programs[k] = wrap