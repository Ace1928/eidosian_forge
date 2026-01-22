import argparse
import importlib
import os
import sys as _sys
import datetime
import parlai
import parlai.utils.logging as logging
from parlai.core.build_data import modelzoo_path
from parlai.core.loader import (
from parlai.tasks.tasks import ids_to_tasks
from parlai.core.opt import Opt
from typing import List, Optional
def _value2argstr(self, value) -> str:
    """
        Reverse-parse an opt value into one interpretable by argparse.
        """
    if isinstance(value, (list, tuple)):
        return ','.join((str(v) for v in value))
    else:
        return str(value)