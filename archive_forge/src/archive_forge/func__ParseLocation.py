from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import list_pager
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
def _ParseLocation(location):
    return resources.REGISTRY.Parse(location, params={'projectsId': properties.VALUES.core.project.GetOrFail}, collection='ml.projects.locations')