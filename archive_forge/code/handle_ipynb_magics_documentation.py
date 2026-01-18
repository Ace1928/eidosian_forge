import ast
import collections
import dataclasses
import secrets
import sys
from functools import lru_cache
from importlib.util import find_spec
from typing import Dict, List, Optional, Tuple
from black.output import out
from black.report import NothingChanged
Look for magics in body of cell.

        For examples,

            !ls
            !!ls
            ?ls
            ??ls

        would (respectively) get transformed to

            get_ipython().system('ls')
            get_ipython().getoutput('ls')
            get_ipython().run_line_magic('pinfo', 'ls')
            get_ipython().run_line_magic('pinfo2', 'ls')

        and we look for instances of any of the latter.
        