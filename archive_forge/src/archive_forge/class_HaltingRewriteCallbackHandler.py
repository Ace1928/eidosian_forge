from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
class HaltingRewriteCallbackHandler(object):
    """Test callback handler for intentionally stopping a rewrite operation."""

    def __init__(self, halt_at_byte):
        self._halt_at_byte = halt_at_byte

    def call(self, total_bytes_rewritten, unused_total_size):
        """Forcibly exits if the operation has passed the halting point."""
        if total_bytes_rewritten >= self._halt_at_byte:
            raise RewriteHaltException('Artificially halting rewrite')