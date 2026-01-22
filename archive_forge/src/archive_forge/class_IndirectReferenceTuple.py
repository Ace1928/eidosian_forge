from __future__ import annotations
import calendar
import codecs
import collections
import mmap
import os
import re
import time
import zlib
from typing import TYPE_CHECKING, Any, List, NamedTuple, Union
class IndirectReferenceTuple(NamedTuple):
    object_id: int
    generation: int