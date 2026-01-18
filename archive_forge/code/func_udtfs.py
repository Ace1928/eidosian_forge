from __future__ import annotations
import itertools
import logging
import typing as t
from collections import defaultdict
from enum import Enum, auto
from sqlglot import exp
from sqlglot.errors import OptimizeError
from sqlglot.helper import ensure_collection, find_new_name, seq_get
@property
def udtfs(self):
    """
        List of "User Defined Tabular Functions" in this scope.

        Returns:
            list[exp.UDTF]: UDTFs
        """
    self._ensure_collected()
    return self._udtfs