import warnings
import weakref
from collections import UserDict
from typing import TYPE_CHECKING, Dict, Iterable, List, Optional, Tuple, Union
import srsly
from ..errors import Errors, Warnings
from .span_group import SpanGroup
A dict-like proxy held by the Doc, to control access to span groups.