from __future__ import annotations
from typing import Optional
def _eval_propagate_not(self):
    return And(*[Not(a) for a in self.args])