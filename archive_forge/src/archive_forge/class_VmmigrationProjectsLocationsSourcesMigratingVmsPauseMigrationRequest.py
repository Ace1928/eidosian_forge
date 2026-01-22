from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class VmmigrationProjectsLocationsSourcesMigratingVmsPauseMigrationRequest(_messages.Message):
    """A VmmigrationProjectsLocationsSourcesMigratingVmsPauseMigrationRequest
  object.

  Fields:
    migratingVm: Required. The name of the MigratingVm.
    pauseMigrationRequest: A PauseMigrationRequest resource to be passed as
      the request body.
  """
    migratingVm = _messages.StringField(1, required=True)
    pauseMigrationRequest = _messages.MessageField('PauseMigrationRequest', 2)