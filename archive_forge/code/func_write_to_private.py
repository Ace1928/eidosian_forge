from __future__ import annotations
import importlib.resources
from pathlib import PurePosixPath, Path
import typing as T
def write_to_private(self, env: 'Environment') -> Path:
    try:
        resource = importlib.resources.files('mesonbuild') / self.path
        if isinstance(resource, Path):
            return resource
    except AttributeError:
        pass
    out_file = Path(env.scratch_dir) / 'data' / self.path.name
    out_file.parent.mkdir(exist_ok=True)
    self.write_once(out_file)
    return out_file