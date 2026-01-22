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
def _temp_multiinput(input: Union[str, os.PathLike, Mapping[str, Any], List[Any], None], base: int=1) -> Iterator[Optional[str]]:
    if isinstance(input, list):
        mother_file = create_named_text_file(dir=_TMPDIR, prefix='', suffix='.json', name_only=True)
        new_files = [os.path.splitext(mother_file)[0] + f'_{i + base}.json' for i in range(len(input))]
        for init, file in zip(input, new_files):
            if isinstance(init, dict):
                write_stan_json(file, init)
            elif isinstance(init, str):
                shutil.copy(init, file)
            else:
                raise ValueError('A list of inits must contain dicts or strings, not' + str(type(init)))
        try:
            yield mother_file
        finally:
            for file in new_files:
                with contextlib.suppress(PermissionError):
                    os.remove(file)
    else:
        yield from _temp_single_json(input)