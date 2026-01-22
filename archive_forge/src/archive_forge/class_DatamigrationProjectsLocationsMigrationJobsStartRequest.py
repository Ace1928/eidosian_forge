from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DatamigrationProjectsLocationsMigrationJobsStartRequest(_messages.Message):
    """A DatamigrationProjectsLocationsMigrationJobsStartRequest object.

  Fields:
    name: Name of the migration job resource to start.
    startMigrationJobRequest: A StartMigrationJobRequest resource to be passed
      as the request body.
  """
    name = _messages.StringField(1, required=True)
    startMigrationJobRequest = _messages.MessageField('StartMigrationJobRequest', 2)