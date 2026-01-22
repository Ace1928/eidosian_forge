from __future__ import annotations
import decimal
import re
from collections.abc import Iterable, Mapping
from typing import TYPE_CHECKING, Any, Callable
def and_condition(self):
    op = self.relation()
    while skip_token(self.tokens, 'word', 'and'):
        op = ('and', (op, self.relation()))
    return op