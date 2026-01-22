from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DatamigrationProjectsLocationsMigrationJobsGenerateSshScriptRequest(_messages.Message):
    """A DatamigrationProjectsLocationsMigrationJobsGenerateSshScriptRequest
  object.

  Fields:
    generateSshScriptRequest: A GenerateSshScriptRequest resource to be passed
      as the request body.
    name: Name of the migration job resource to generate the SSH script.
  """
    generateSshScriptRequest = _messages.MessageField('GenerateSshScriptRequest', 1)
    name = _messages.StringField(2, required=True)