from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import itertools
from apitools.base.py import list_pager
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.core import resources
def ListOccurrencesWithFiltersV1beta1(project, filters):
    """List occurrences for resources in a project with multiple filters."""
    results = [ListOccurrencesV1beta1(project, f) for f in filters]
    return itertools.chain(*results)