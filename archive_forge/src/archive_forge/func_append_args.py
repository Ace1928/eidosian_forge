from __future__ import annotations
from ..mesonlib import MesonException, OptionKey
from .. import mlog
from pathlib import Path
import typing as T
def append_args(self, lang: str, args: T.List[str]) -> None:
    if lang not in self.lang_args:
        self.lang_args[lang] = []
    self.lang_args[lang] += args