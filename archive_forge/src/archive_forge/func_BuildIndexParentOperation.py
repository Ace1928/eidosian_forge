from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.core import resources
def BuildIndexParentOperation(project_id, location_id, index_id, operation_id):
    """Build multi-parent operation."""
    return ParseIndexOperation('projects/{}/locations/{}/indexes/{}/operations/{}'.format(project_id, location_id, index_id, operation_id))