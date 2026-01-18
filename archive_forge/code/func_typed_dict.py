import sys
from contextlib import contextmanager
from typing import (
from functools import partial
from IPython.core.guarded_eval import (
from IPython.testing import decorators as dec
import pytest
def typed_dict(self) -> Movie:
    return {'name': 'The Matrix', 'year': 1999}