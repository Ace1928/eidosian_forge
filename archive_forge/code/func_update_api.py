from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import abc
from googlecloudsdk.calliope import base as calliope_base
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import yaml
from googlecloudsdk.core import yaml_validator
from googlecloudsdk.core.util import files
import six
def update_api(self, api_data):
    """Updates an API's data with the provided api data.

    Args:
      api_data: API Data to update the api with. Must be provided as an ApiData
      object.

    Raises:
      ApiNotFoundError: Api to be updated does not exist.
      UnwrappedDataException: API data attempting to be added without being
        wrapped in an ApiData wrapper object.
    """
    if not isinstance(api_data, ApiData):
        raise UnwrappedDataException('Api', api_data)
    if api_data.get_api_name() not in self._resource_map_data:
        raise ApiNotFoundError(api_data.get_api_name())
    else:
        self._resource_map_data.update(api_data.to_dict())