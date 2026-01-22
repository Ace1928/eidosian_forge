import re
from collections import defaultdict
from typing import Any, Counter, Dict, List, Match, Optional, Sequence, Set, Tuple
import yaml
from torchgen.api import cpp
from torchgen.api.autograd import (
from torchgen.api.types import (
from torchgen.context import with_native_function
from torchgen.gen import get_grouped_by_view_native_functions, parse_native_yaml
from torchgen.model import (
from torchgen.utils import concatMap, IDENT_REGEX, split_name_params
from torchgen.yaml_utils import YamlLoader
def _create_op_prefix(name: str) -> str:
    """Takes a native function name converts to a op prefix name.

    Note that the "name" parameter must be the native function name
    without the optional variant suffix, so "add" instead of
    "add.out".

    OP names correspond to classes, hence the change to title case.

    Example::
    >>> _create_op_prefix('add')
    'AddBackward'
    """
    camel_case = ''.join([p.title() for p in name.split('_')])
    return (camel_case + 'Backward').replace('ForwardBackward', 'Backward')