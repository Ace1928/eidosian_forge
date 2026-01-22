from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class BinarySourceInfo(_messages.Message):
    """A BinarySourceInfo object.

  Fields:
    binaryVersion: The binary package. This is significant when the source is
      different than the binary itself. Historically if they've differed,
      we've stored the name of the source and its version in the
      package/version fields, but we should also store the binary package
      info, as that's what's actually installed. See b/175908657#comment15.
    sourceVersion: The source package. Similar to the above, this is
      significant when the source is different than the binary itself. Since
      the top-level package/version fields are based on an if/else, we need a
      separate field for both binary and source if we want to know
      definitively where the data is coming from.
  """
    binaryVersion = _messages.MessageField('PackageVersion', 1)
    sourceVersion = _messages.MessageField('PackageVersion', 2)