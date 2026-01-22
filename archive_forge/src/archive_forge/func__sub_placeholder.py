import os
from typing import Dict, Type
def _sub_placeholder(self, sql):
    """Format the argument placeholders for sqlite (PRIVATE)."""
    return sql.replace('%s', '?')