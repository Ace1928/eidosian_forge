from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import list_pager
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.command_lib.ai import constants
from googlecloudsdk.command_lib.ai import errors
from googlecloudsdk.command_lib.util.args import labels_util
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
def _ParseIndex(index_id, location_id):
    """Parses a index ID into a index resource object."""
    return resources.REGISTRY.Parse(index_id, params={'locationsId': location_id, 'projectsId': properties.VALUES.core.project.GetOrFail}, collection='aiplatform.projects.locations.indexes')