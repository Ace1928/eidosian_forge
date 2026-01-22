import logging
import os
import re
import site
import sys
from typing import List, Optional
def _running_under_venv() -> bool:
    """Checks if sys.base_prefix and sys.prefix match.

    This handles PEP 405 compliant virtual environments.
    """
    return sys.prefix != getattr(sys, 'base_prefix', sys.prefix)