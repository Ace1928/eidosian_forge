import itertools
import os
import struct
from copy import copy
from io import BytesIO
from typing import Any, Tuple
from zlib import adler32
from ..lazy_import import lazy_import
import fastbencode as bencode
from breezy import (
from breezy.bzr import (
from .. import errors
from .. import graph as _mod_graph
from .. import osutils
from .. import transport as _mod_transport
from ..registry import Registry
from ..textmerge import TextMerge
from . import index
def weave_merge(self, plan, a_marker=TextMerge.A_MARKER, b_marker=TextMerge.B_MARKER):
    return PlanWeaveMerge(plan, a_marker, b_marker).merge_lines()[0]