from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GdceCluster(_messages.Message):
    """Gdce cluster information.

  Fields:
    gdceCluster: Required. Gdce cluster resource id.
  """
    gdceCluster = _messages.StringField(1)