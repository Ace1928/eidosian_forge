from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class InventoryWindowsUpdatePackageWindowsUpdateCategory(_messages.Message):
    """Categories specified by the Windows Update.

  Fields:
    id: The identifier of the windows update category.
    name: The name of the windows update category.
  """
    id = _messages.StringField(1)
    name = _messages.StringField(2)