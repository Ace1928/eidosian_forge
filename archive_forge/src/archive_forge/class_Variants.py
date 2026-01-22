from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class Variants(_messages.Message):
    """Variants is the configuration for how to read the repository to find
  variants.

  Fields:
    directories: Required. directories is the set of directories to use to
      select variants.
  """
    directories = _messages.MessageField('Directories', 1)