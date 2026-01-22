import sys
from contextlib import contextmanager
from typing import (
from functools import partial
from IPython.core.guarded_eval import (
from IPython.testing import decorators as dec
import pytest
def annotated_self(self) -> Annotated[Self, 'self with metadata']:
    self._metadata = 'test'
    return self