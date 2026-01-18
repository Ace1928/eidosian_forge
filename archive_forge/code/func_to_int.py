from __future__ import annotations
from typing import Any
import operator
from sympy.polys.polyutils import PicklableWithSlots
from sympy.polys.polyerrors import CoercionFailed
from sympy.polys.domains.domainelement import DomainElement
from sympy.utilities import public
def to_int(self):
    if self.sym:
        if self.val <= self.mod // 2:
            return self.val
        else:
            return self.val - self.mod
    else:
        return self.val