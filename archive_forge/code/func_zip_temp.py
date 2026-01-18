import os
import re
import tempfile
import zipfile
from contextlib import contextmanager
from pathlib import Path, PurePosixPath
from typing import Any, Iterator, List, Tuple
import fsspec
import fsspec.core as fc
from fsspec import AbstractFileSystem
from fsspec.implementations.local import LocalFileSystem
@contextmanager
def zip_temp(fobj: Any) -> Iterator[str]:
    """Zip a temporary directory to a file object.

    :param fobj: the file path or file object

    .. admonition:: Examples

        .. code-block:: python

            from fugue_ml.utils.io import zip_temp
            from io import BytesIO

            bio = BytesIO()
            with zip_temp(bio) as tmpdir:
                # do something with tmpdir (string)
    """
    if isinstance(fobj, str):
        with fsspec.open(fobj, 'wb', create_dir=True) as f:
            with zip_temp(f) as tmpdir:
                yield tmpdir
    else:
        with tempfile.TemporaryDirectory() as tmpdirname:
            yield tmpdirname
            with zipfile.ZipFile(fobj, 'w', zipfile.ZIP_DEFLATED, allowZip64=True) as zf:
                for root, _, filenames in os.walk(tmpdirname):
                    for name in filenames:
                        file_path = os.path.join(root, name)
                        rel_dir = os.path.relpath(root, tmpdirname)
                        rel_name = os.path.normpath(os.path.join(rel_dir, name))
                        zf.write(file_path, rel_name)