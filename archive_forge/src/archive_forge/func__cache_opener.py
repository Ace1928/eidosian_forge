import contextlib
import errno
import hashlib
import os
import shutil
import subprocess
import sys
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import IO, TYPE_CHECKING, ContextManager, Generator, Optional, Tuple
import wandb
from wandb import env, util
from wandb.errors import term
from wandb.sdk.lib.filesystem import files_in
from wandb.sdk.lib.hashutil import B64MD5, ETag, b64_to_hex_id
from wandb.sdk.lib.paths import FilePathStr, StrPath, URIStr
def _cache_opener(self, path: Path, size: int) -> 'Opener':

    @contextlib.contextmanager
    def helper(mode: str='w') -> Generator[IO, None, None]:
        if 'a' in mode:
            raise ValueError('Appending to cache files is not supported')
        self._reserve_space(size)
        temp_file = NamedTemporaryFile(dir=self._temp_dir, mode=mode, delete=False)
        try:
            yield temp_file
            temp_file.close()
            os.chmod(temp_file.name, 438 & ~self._sys_umask)
            path.parent.mkdir(parents=True, exist_ok=True)
            os.replace(temp_file.name, path)
        except Exception:
            os.remove(temp_file.name)
            raise
    return helper