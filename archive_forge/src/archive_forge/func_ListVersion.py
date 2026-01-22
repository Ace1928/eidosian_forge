from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import list_pager
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.command_lib.ai import constants
def ListVersion(self, model_ref=None, limit=None):
    """List all model versions of the given model.

    Args:
      model_ref: The resource reference for a given model. None if model
        resource reference is not provided.
      limit: int, The maximum number of records to yield. None if all available
        records should be yielded.

    Returns:
      Response from calling list model versions with request containing given
      model and limit.
    """
    return list_pager.YieldFromList(self._service, self.messages.AiplatformProjectsLocationsModelsListVersionsRequest(name=model_ref.RelativeName()), method='ListVersions', field='models', batch_size_attribute='pageSize', limit=limit)