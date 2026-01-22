import collections
import contextlib
import dataclasses
import os
import shutil
import tempfile
import textwrap
import time
from typing import cast, Any, DefaultDict, Dict, Iterable, Iterator, List, Optional, Tuple
import uuid
import torch
def _make_temp_dir(prefix: Optional[str]=None, gc_dev_shm: bool=False) -> str:
    """Create a temporary directory. The caller is responsible for cleanup.

    This function is conceptually similar to `tempfile.mkdtemp`, but with
    the key additional feature that it will use shared memory if the
    `BENCHMARK_USE_DEV_SHM` environment variable is set. This is an
    implementation detail, but an important one for cases where many Callgrind
    measurements are collected at once. (Such as when collecting
    microbenchmarks.)

    This is an internal utility, and is exported solely so that microbenchmarks
    can reuse the util.
    """
    use_dev_shm: bool = (os.getenv('BENCHMARK_USE_DEV_SHM') or '').lower() in ('1', 'true')
    if use_dev_shm:
        root = '/dev/shm/pytorch_benchmark_utils'
        assert os.name == 'posix', f'tmpfs (/dev/shm) is POSIX only, current platform is {os.name}'
        assert os.path.exists('/dev/shm'), 'This system does not appear to support tmpfs (/dev/shm).'
        os.makedirs(root, exist_ok=True)
        if gc_dev_shm:
            for i in os.listdir(root):
                owner_file = os.path.join(root, i, 'owner.pid')
                if not os.path.exists(owner_file):
                    continue
                with open(owner_file) as f:
                    owner_pid = int(f.read())
                if owner_pid == os.getpid():
                    continue
                try:
                    os.kill(owner_pid, 0)
                except OSError:
                    print(f'Detected that {os.path.join(root, i)} was orphaned in shared memory. Cleaning up.')
                    shutil.rmtree(os.path.join(root, i))
    else:
        root = tempfile.gettempdir()
    name = f'{prefix or tempfile.gettempprefix()}__{int(time.time())}__{uuid.uuid4()}'
    path = os.path.join(root, name)
    os.makedirs(path, exist_ok=False)
    if use_dev_shm:
        with open(os.path.join(path, 'owner.pid'), 'w') as f:
            f.write(str(os.getpid()))
    return path