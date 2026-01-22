from __future__ import annotations
from copy import copy
from typing import Any
from tomlkit.exceptions import ParseError
from tomlkit.exceptions import UnexpectedCharError
from tomlkit.toml_char import TOMLChar
def _to_linecol(self) -> tuple[int, int]:
    cur = 0
    for i, line in enumerate(self.splitlines()):
        if cur + len(line) + 1 > self.idx:
            return (i + 1, self.idx - cur)
        cur += len(line) + 1
    return (len(self.splitlines()), 0)