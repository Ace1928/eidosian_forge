from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
class Preconditions(object):
    """Preconditions class for specifying preconditions to cloud API requests."""

    def __init__(self, gen_match=None, meta_gen_match=None):
        """Instantiates a Preconditions object.

    Args:
      gen_match: Perform request only if generation of target object
                 matches the given integer. Ignored for bucket requests.
      meta_gen_match: Perform request only if metageneration of target
                      object/bucket matches the given integer.
    """
        self.gen_match = gen_match
        self.meta_gen_match = meta_gen_match