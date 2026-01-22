import contextlib
import os
import platform
import re
import shutil
import tempfile
from typing import Any, Iterator, List, Mapping, Optional, Tuple, Union
from cmdstanpy import _TMPDIR
from .json import write_stan_json
from .logging import get_logger
def _temp_single_json(data: Union[str, os.PathLike, Mapping[str, Any], None]) -> Iterator[Optional[str]]:
    """Context manager for json files."""
    if data is None:
        yield None
        return
    if isinstance(data, (str, os.PathLike)):
        yield str(data)
        return
    data_file = create_named_text_file(dir=_TMPDIR, prefix='', suffix='.json')
    get_logger().debug('input tempfile: %s', data_file)
    write_stan_json(data_file, data)
    try:
        yield data_file
    finally:
        with contextlib.suppress(PermissionError):
            os.remove(data_file)