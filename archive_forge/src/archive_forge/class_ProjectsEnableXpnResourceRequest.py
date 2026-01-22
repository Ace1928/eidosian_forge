from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ProjectsEnableXpnResourceRequest(_messages.Message):
    """A ProjectsEnableXpnResourceRequest object.

  Fields:
    xpnResource: Service resource (a.k.a service project) ID.
  """
    xpnResource = _messages.MessageField('XpnResourceId', 1)