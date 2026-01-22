from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CloudSQLConfig(_messages.Message):
    """Cloud SQL configuration.

  Fields:
    service: Peering service used for peering with the Cloud SQL project.
    umbrellaNetwork: The name of the umbrella network in the Cloud SQL
      umbrella project.
    umbrellaProject: The project number of the Cloud SQL umbrella project.
  """
    service = _messages.StringField(1)
    umbrellaNetwork = _messages.StringField(2)
    umbrellaProject = _messages.IntegerField(3)