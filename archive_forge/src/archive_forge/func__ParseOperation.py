from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.certificate_manager import api_client
from googlecloudsdk.api_lib.certificate_manager import operations
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
def _ParseOperation(operation):
    return _GetRegistry().Parse(operation, params={'projectsId': _PROJECT, 'locationsId': 'global'}, collection=_OPERATIONS_COLLECTION)