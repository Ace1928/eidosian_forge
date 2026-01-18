from __future__ import annotations
from enum import Enum
from typing import TYPE_CHECKING, Any, Dict, List, NamedTuple, Optional
Add a ChildRecord to self._children. If `index` is specified, replace
        the existing record at that index. Otherwise, append the record to the
        end of the list.

        Return the index of the added record.
        