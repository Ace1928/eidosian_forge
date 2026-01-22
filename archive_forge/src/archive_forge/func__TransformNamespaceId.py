from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.datastore import constants
from googlecloudsdk.api_lib.datastore import util
def _TransformNamespaceId(namespace_id):
    """Transforms client namespace conventions into server conventions."""
    if namespace_id == constants.DEFAULT_NAMESPACE:
        return ''
    return namespace_id