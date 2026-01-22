from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GooglePrivacyDlpV2PartitionId(_messages.Message):
    """Datastore partition ID. A partition ID identifies a grouping of
  entities. The grouping is always by project and namespace, however the
  namespace ID may be empty. A partition ID contains several dimensions:
  project ID and namespace ID.

  Fields:
    namespaceId: If not empty, the ID of the namespace to which the entities
      belong.
    projectId: The ID of the project to which the entities belong.
  """
    namespaceId = _messages.StringField(1)
    projectId = _messages.StringField(2)