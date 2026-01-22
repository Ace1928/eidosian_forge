from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.core.cache import cache_update_ops
def AddFlatten(self, flatten):
    """Adds a flatten to the display info, newer info takes precedence.

    Args:
      flatten: [str], The list of flatten strings to use by default; e.g.
        ['fieldA.fieldB', 'fieldC.fieldD']. Flatten strings provided by
        args.flatten take precedence if the --flatten flag is specified.
    """
    if flatten:
        self._flatten = flatten