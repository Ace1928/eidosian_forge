from __future__ import annotations
from contextlib import suppress
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Any
import json
import os
import re
import subprocess
import sys
import unittest
from attrs import field, frozen
from referencing import Registry
import referencing.jsonschema
from jsonschema.validators import _VALIDATORS
import jsonschema
def _cases_in(self, paths: Iterable[Path]) -> Iterable[_Case]:
    for path in paths:
        for case in json.loads(path.read_text(encoding='utf-8')):
            yield _Case.from_dict(case, version=self, subject=path.stem, remotes=self._remotes)