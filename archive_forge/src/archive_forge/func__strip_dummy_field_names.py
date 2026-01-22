import contextlib
import functools
import re
import textwrap
from typing import Iterable, List, Sequence, Tuple, Type
from typing_extensions import Literal, get_args, get_origin
from . import _resolver
def _strip_dummy_field_names(parts: Iterable[str]) -> Iterable[str]:
    return filter(lambda name: len(name) > 0 and name != dummy_field_name, parts)