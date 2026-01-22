from dataclasses import dataclass
import logging
import os
import sys
from typing import List, Tuple, Optional, TYPE_CHECKING
def _is_in_test():
    global _in_test
    if _in_test is None:
        _in_test = any((env_var in os.environ for env_var in ('PYTEST_CURRENT_TEST', 'BUILDKITE')))
    return _in_test