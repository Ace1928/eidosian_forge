from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.datastore import constants
from googlecloudsdk.api_lib.datastore import util
def _MakeEntityFilter(namespaces, kinds):
    """Creates an entity filter for the given namespaces and kinds.

  Args:
    namespaces: a string list of the namespaces to include in the filter.
    kinds: a string list of the kinds to include in the filter.
  Returns:
    a GetMessages().EntityFilter (proto).
  """
    namespaces = namespaces or []
    namespaces = [_TransformNamespaceId(namespace) for namespace in namespaces]
    return util.GetMessages().GoogleDatastoreAdminV1EntityFilter(kinds=kinds or [], namespaceIds=namespaces)