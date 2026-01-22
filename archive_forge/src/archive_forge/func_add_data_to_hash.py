imports working.
from __future__ import annotations
import glob
import hashlib
import os.path
from typing import Callable, Iterable
from coverage.exceptions import CoverageException, NoDataError
from coverage.files import PathAliases
from coverage.misc import Hasher, file_be_gone, human_sorted, plural
from coverage.sqldata import CoverageData
def add_data_to_hash(data: CoverageData, filename: str, hasher: Hasher) -> None:
    """Contribute `filename`'s data to the `hasher`.

    `hasher` is a `coverage.misc.Hasher` instance to be updated with
    the file's data.  It should only get the results data, not the run
    data.

    """
    if data.has_arcs():
        hasher.update(sorted(data.arcs(filename) or []))
    else:
        hasher.update(sorted_lines(data, filename))
    hasher.update(data.file_tracer(filename))