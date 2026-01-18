from __future__ import unicode_literals
from itertools import tee, chain
import re
import copy
def to_last(self, doc):
    """Resolves ptr until the last step, returns (sub-doc, last-step)"""
    if not self.parts:
        return (doc, None)
    for part in self.parts[:-1]:
        doc = self.walk(doc, part)
    return (doc, JsonPointer.get_part(doc, self.parts[-1]))