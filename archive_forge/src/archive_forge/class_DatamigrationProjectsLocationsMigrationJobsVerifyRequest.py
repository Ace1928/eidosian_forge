from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DatamigrationProjectsLocationsMigrationJobsVerifyRequest(_messages.Message):
    """A DatamigrationProjectsLocationsMigrationJobsVerifyRequest object.

  Fields:
    name: Name of the migration job resource to verify.
    verifyMigrationJobRequest: A VerifyMigrationJobRequest resource to be
      passed as the request body.
  """
    name = _messages.StringField(1, required=True)
    verifyMigrationJobRequest = _messages.MessageField('VerifyMigrationJobRequest', 2)