from gitdb.util import bin_to_hex
from gitdb.fun import (
def _set_binsha(self, binsha):
    self[0] = binsha