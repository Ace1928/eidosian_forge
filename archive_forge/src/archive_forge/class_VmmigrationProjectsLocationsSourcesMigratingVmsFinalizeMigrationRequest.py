from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class VmmigrationProjectsLocationsSourcesMigratingVmsFinalizeMigrationRequest(_messages.Message):
    """A
  VmmigrationProjectsLocationsSourcesMigratingVmsFinalizeMigrationRequest
  object.

  Fields:
    finalizeMigrationRequest: A FinalizeMigrationRequest resource to be passed
      as the request body.
    migratingVm: Required. The name of the MigratingVm.
  """
    finalizeMigrationRequest = _messages.MessageField('FinalizeMigrationRequest', 1)
    migratingVm = _messages.StringField(2, required=True)