from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ResourceFilter(_messages.Message):
    """ResourceFilter specifies matching criteria to limit the scope of a
  change to a specific set of kubernetes resources that are selected for
  restoration from a backup.

  Fields:
    groupKinds: Optional. (Filtering parameter) Any resource subject to
      transformation must belong to one of the listed "types". If this field
      is not provided, no type filtering will be performed (all resources of
      all types matching previous filtering parameters will be candidates for
      transformation).
    jsonPath: Optional. This is a [JSONPath] (https://github.com/json-
      path/JsonPath/blob/master/README.md) expression that matches specific
      fields of candidate resources and it operates as a filtering parameter
      (resources that are not matched with this expression will not be
      candidates for transformation).
    namespaces: Optional. (Filtering parameter) Any resource subject to
      transformation must be contained within one of the listed Kubernetes
      Namespace in the Backup. If this field is not provided, no namespace
      filtering will be performed (all resources in all Namespaces, including
      all cluster-scoped resources, will be candidates for transformation).
  """
    groupKinds = _messages.MessageField('GroupKind', 1, repeated=True)
    jsonPath = _messages.StringField(2)
    namespaces = _messages.StringField(3, repeated=True)