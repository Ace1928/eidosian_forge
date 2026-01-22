from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class VmmigrationProjectsLocationsSourcesMigratingVmsStartMigrationRequest(_messages.Message):
    """A VmmigrationProjectsLocationsSourcesMigratingVmsStartMigrationRequest
  object.

  Fields:
    migratingVm: Required. The name of the MigratingVm.
    startMigrationRequest: A StartMigrationRequest resource to be passed as
      the request body.
  """
    migratingVm = _messages.StringField(1, required=True)
    startMigrationRequest = _messages.MessageField('StartMigrationRequest', 2)