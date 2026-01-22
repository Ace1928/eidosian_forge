from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class BootImage(_messages.Message):
    """Definition of the boot image used by the Runtime. Used to facilitate
  runtime upgradeability.
  """