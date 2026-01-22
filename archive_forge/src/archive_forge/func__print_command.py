from .. import (
import stat
def _print_command(self, cmd):
    """Wrapper to avoid adding unnecessary blank lines."""
    text = bytes(cmd)
    self.outf.write(text)
    if not text.endswith(b'\n'):
        self.outf.write(b'\n')