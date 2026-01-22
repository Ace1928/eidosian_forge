from __future__ import annotations
import os
import json
import re
import sys
import shutil
import typing as T
from collections import defaultdict
from pathlib import Path
from . import mlog
from . import mesonlib
from .mesonlib import MesonException, RealPathAction, join_args, setup_vsenv
from mesonbuild.environment import detect_ninja
from mesonbuild.coredata import UserArrayOption
from mesonbuild import build
class ParsedTargetName:
    full_name = ''
    base_name = ''
    name = ''
    type = ''
    path = ''
    suffix = ''

    def __init__(self, target: str):
        self.full_name = target
        split = target.rsplit(':', 1)
        if len(split) > 1:
            self.type = split[1]
            if not self._is_valid_type(self.type):
                raise MesonException(f"Can't invoke target `{target}`: unknown target type: `{self.type}`")
        split = split[0].rsplit('/', 1)
        if len(split) > 1:
            self.path = split[0]
            self.name = split[1]
        else:
            self.name = split[0]
        split = self.name.rsplit('.', 1)
        if len(split) > 1:
            self.base_name = split[0]
            self.suffix = split[1]
        else:
            self.base_name = split[0]

    @staticmethod
    def _is_valid_type(type: str) -> bool:
        allowed_types = {'executable', 'static_library', 'shared_library', 'shared_module', 'custom', 'alias', 'run', 'jar'}
        return type in allowed_types