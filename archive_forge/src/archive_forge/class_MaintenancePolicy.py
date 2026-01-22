from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class MaintenancePolicy(_messages.Message):
    """MaintenancePolicy defines the maintenance policy to be used for the
  cluster.

  Fields:
    resourceVersion: A hash identifying the version of this policy, so that
      updates to fields of the policy won't accidentally undo intermediate
      changes (and so that users of the API unaware of some fields won't
      accidentally remove other fields). Make a `get()` request to the cluster
      to get the current resource version and include it with requests to set
      the policy.
    window: Specifies the maintenance window in which maintenance may be
      performed.
  """
    resourceVersion = _messages.StringField(1)
    window = _messages.MessageField('MaintenanceWindow', 2)