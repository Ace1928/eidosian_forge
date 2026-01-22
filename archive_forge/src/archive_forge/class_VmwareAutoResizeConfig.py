from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class VmwareAutoResizeConfig(_messages.Message):
    """Represents auto resizing configurations for the VMware user cluster.

  Fields:
    enabled: Whether to enable controle plane node auto resizing.
  """
    enabled = _messages.BooleanField(1)