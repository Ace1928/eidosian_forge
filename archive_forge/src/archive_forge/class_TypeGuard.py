from __future__ import annotations
import sys
class TypeGuard:

    def __class_getitem__(cls, item: Any) -> type[bool]:
        return bool