import html.entities as htmlentitydefs
import re
import warnings
from ast import literal_eval
from collections import defaultdict
from enum import Enum
from io import StringIO
from typing import Any, NamedTuple
import networkx as nx
from networkx.exception import NetworkXError
from networkx.utils import open_file
def unexpected(curr_token, expected):
    category, value, lineno, pos = curr_token
    value = repr(value) if value is not None else 'EOF'
    raise NetworkXError(f'expected {expected}, found {value} at ({lineno}, {pos})')