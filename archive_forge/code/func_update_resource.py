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
def update_resource(self, resource_data):
    if not isinstance(resource_data, ResourceData):
        raise UnwrappedDataException('Resource', resource_data)
    elif resource_data.get_resource_name() not in self._api_data:
        raise ResourceNotFoundError(resource_data.get_resource_name())
    else:
        self._api_data.update(resource_data.to_dict())