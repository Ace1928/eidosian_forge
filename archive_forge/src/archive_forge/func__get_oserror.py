from __future__ import annotations
import io
import itertools
import struct
import sys
from typing import Any, NamedTuple
from . import Image
from ._deprecate import deprecate
from ._util import is_path
def _get_oserror(error, *, encoder):
    try:
        msg = Image.core.getcodecstatus(error)
    except AttributeError:
        msg = ERRORS.get(error)
    if not msg:
        msg = f'{('encoder' if encoder else 'decoder')} error {error}'
    msg += f' when {('writing' if encoder else 'reading')} image file'
    return OSError(msg)