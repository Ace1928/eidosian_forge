import json
from typing import IO, Any, Tuple, List
from .parser import Parser
from .symbols import (
Decoder for the avro JSON format.

    NOTE: All attributes and methods on this class should be considered
    private.

    Parameters
    ----------
    fo
        File-like object to reader from

    