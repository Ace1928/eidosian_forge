from __future__ import annotations
from typing import Callable, Optional
from collections import OrderedDict
import os
import re
import subprocess
from .util import (
def _mk_flag_filter(cmplr_name):
    not_welcome = {'g++': ('Wimplicit-interface',)}
    if cmplr_name in not_welcome:

        def fltr(x):
            for nw in not_welcome[cmplr_name]:
                if nw in x:
                    return False
            return True
    else:

        def fltr(x):
            return True
    return fltr