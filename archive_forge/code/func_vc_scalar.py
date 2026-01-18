import base64
import numbers
import textwrap
import uuid
from importlib import import_module
import copy
import io
import re
import sys
import warnings
from _plotly_utils.optional_imports import get_module
def vc_scalar(self, v):
    if isinstance(v, str):
        v = v.strip()
    if v in self.extras:
        return v
    if not isinstance(v, str):
        return None
    split_vals = [e.strip() for e in re.split('[,+]', v)]
    if all((f in self.flags for f in split_vals)):
        return '+'.join(split_vals)
    else:
        return None