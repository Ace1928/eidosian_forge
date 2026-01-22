from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AutoAccept(_messages.Message):
    """The auto-accept setting for a group controls whether proposed spokes are
  automatically attached to the hub. If auto-accept is enabled, the spoke
  immediately is attached to the hub and becomes part of the group. In this
  case, the new spoke is in the ACTIVE state. If auto-accept is disabled, the
  spoke goes to the INACTIVE state, and it must be reviewed and accepted by a
  hub administrator.

  Fields:
    autoAcceptProjects: A list of project ids or project numbers for which you
      want to enable auto-accept. The auto-accept setting is applied to spokes
      being created or updated in these projects.
  """
    autoAcceptProjects = _messages.StringField(1, repeated=True)