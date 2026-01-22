import sys, os
import textwrap
def _check_dest(self):
    takes_value = self.action in self.STORE_ACTIONS or self.type is not None
    if self.dest is None and takes_value:
        if self._long_opts:
            self.dest = self._long_opts[0][2:].replace('-', '_')
        else:
            self.dest = self._short_opts[0][1]