import contextlib
import dataclasses
import math
import textwrap
from typing import Any, Dict, Optional
import torch
from torch import inf
def _val_formatter(val, formatter1=formatter1, formatter2=formatter2):
    if formatter2 is not None:
        real_str = formatter1.format(val.real)
        imag_str = (formatter2.format(val.imag) + 'j').lstrip()
        if imag_str[0] == '+' or imag_str[0] == '-':
            return real_str + imag_str
        else:
            return real_str + '+' + imag_str
    else:
        return formatter1.format(val)