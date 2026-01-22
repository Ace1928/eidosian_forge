from __future__ import annotations
import dataclasses
import typing as T
from .. import build, mesonlib
from ..build import IncludeDirs
from ..interpreterbase.decorators import noKwargs, noPosargs
from ..mesonlib import relpath, HoldableObject, MachineChoice
from ..programs import ExternalProgram
def add_language(self, lang: str, for_machine: MachineChoice) -> None:
    self._interpreter.add_languages([lang], True, for_machine)