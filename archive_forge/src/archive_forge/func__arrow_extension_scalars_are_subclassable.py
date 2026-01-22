import itertools
import json
import sys
from typing import Iterable, Optional, Tuple, List, Sequence, Union
from pkg_resources._vendor.packaging.version import parse as parse_version
import numpy as np
import pyarrow as pa
from ray.air.util.tensor_extensions.utils import (
from ray._private.utils import _get_pyarrow_version
from ray.util.annotations import PublicAPI
def _arrow_extension_scalars_are_subclassable():
    """
    Whether Arrow ExtensionScalars support subclassing in the current pyarrow version.

    This returns True if the pyarrow version is 9.0.0+, or if the pyarrow version is
    unknown.
    """
    return PYARROW_VERSION is None or PYARROW_VERSION >= MIN_PYARROW_VERSION_SCALAR_SUBCLASS