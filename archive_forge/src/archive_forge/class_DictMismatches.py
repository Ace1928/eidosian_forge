from ..helpers import (
from ._higherorder import (
from ._impl import Matcher, Mismatch
class DictMismatches(Mismatch):
    """A mismatch with a dict of child mismatches."""

    def __init__(self, mismatches, details=None):
        super().__init__(None, details=details)
        self.mismatches = mismatches

    def describe(self):
        lines = ['{']
        lines.extend([f'  {key!r}: {mismatch.describe()},' for key, mismatch in sorted(self.mismatches.items())])
        lines.append('}')
        return '\n'.join(lines)