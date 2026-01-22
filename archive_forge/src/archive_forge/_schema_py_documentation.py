import hashlib
from io import StringIO
import math
from os import path
from copy import deepcopy
import re
from typing import Tuple, Set, Optional, List, Any
from .types import DictSchema, Schema, NamedSchemas
from .repository import (
from .const import AVRO_TYPES
from ._schema_common import (
Returns a string represening a fingerprint/hash of the parsing canonical
    form of a schema.

    For more details on the fingerprint, see here:
    https://avro.apache.org/docs/current/spec.html#schema_fingerprints

    Parameters
    ----------
    parsing_canonical_form
        The parsing canonical form of a schema
    algorithm
        The hashing algorithm

    