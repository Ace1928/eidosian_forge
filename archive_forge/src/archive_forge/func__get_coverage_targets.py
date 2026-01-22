from __future__ import annotations
import collections.abc as c
import os
import json
import typing as t
from ...target import (
from ...io import (
from ...util import (
from ...util_common import (
from ...executor import (
from ...data import (
from ...host_configs import (
from ...provisioning import (
from . import (
def _get_coverage_targets(args: CoverageCombineConfig, walk_func: c.Callable) -> list[tuple[str, int]]:
    """Return a list of files to cover and the number of lines in each file, using the given function as the source of the files."""
    sources = []
    if args.all or args.stub:
        for target in walk_func(include_symlinks=False):
            target_path = os.path.abspath(target.path)
            target_lines = len(read_text_file(target_path).splitlines())
            sources.append((target_path, target_lines))
        sources.sort()
    return sources