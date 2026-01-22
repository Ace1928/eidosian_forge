from typing import Any
import sys
from typing import _type_check  # type: ignore
class _FlexibleAliasClsApplied:

    def __init__(self, val):
        self.val = val

    def __getitem__(self, args):
        return self.val