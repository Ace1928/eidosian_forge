from __future__ import annotations
from collections import namedtuple
import re
def _get_dev_status(self) -> str:
    """Get development status string."""
    return DEV_STATUS[self.release]