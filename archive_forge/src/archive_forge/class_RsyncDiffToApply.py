from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
class RsyncDiffToApply(object):
    """Class that encapsulates info needed to apply diff for one object."""

    def __init__(self, src_url_str, dst_url_str, src_posix_attrs, diff_action, copy_size):
        """Constructor.

    Args:
      src_url_str: (str or None) The source URL string, or None if diff_action
          is REMOVE.
      dst_url_str: (str) The destination URL string.
      src_posix_attrs: (posix_util.POSIXAttributes) The source POSIXAttributes.
      diff_action: (DiffAction) DiffAction to be applied.
      copy_size: (int or None) The amount of bytes to copy, or None if
          diff_action is REMOVE.
    """
        self.src_url_str = src_url_str
        self.dst_url_str = dst_url_str
        self.src_posix_attrs = src_posix_attrs
        self.diff_action = diff_action
        self.copy_size = copy_size