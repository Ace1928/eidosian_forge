from __future__ import annotations
import importlib.resources
from pathlib import PurePosixPath, Path
import typing as T
def write_once(self, path: Path) -> None:
    if not path.exists():
        data = importlib.resources.read_text(('mesonbuild' / self.path.parent).as_posix().replace('/', '.'), self.path.name, encoding='utf-8')
        path.write_text(data, encoding='utf-8')