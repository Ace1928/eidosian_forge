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
class ApiData(object):
    """Data wrapper for an API object in the Resource Map metadata file.

  Attributes:
    _api_name: Name of the API.
    _api_data: Dict of resources and associated metadata constituting the api.
  """

    def __init__(self, api_name, api_data):
        self._api_name = api_name
        self._api_data = api_data

    def __getattr__(self, resource_name):
        """Returns the specified resource's data wrapped in a ResourceData object."""
        if resource_name.startswith('_'):
            raise PrivateAttributeNotFoundError('ApiData', resource_name)
        return ResourceData(resource_name, self._api_name, self._api_data[resource_name])

    def __contains__(self, resource_name):
        return resource_name in self._api_data

    def __iter__(self):
        """Yields ResourceData wrapper objects for each API in _resource_map_data."""
        for resource_name, resource_data in self._api_data.items():
            yield ResourceData(resource_name, self._api_name, resource_data)

    def __repr__(self):
        return repr(self._api_data)

    def __eq__(self, other):
        return self.to_dict() == other.to_dict()

    def to_str(self):
        return six.text_type(self.to_dict())

    def to_dict(self):
        return {self.get_api_name(): self._api_data}

    def get_api_name(self):
        return six.text_type(self._api_name)

    def get_resource(self, resource_name):
        """Returns the data for the specified resource in a ResourceData object."""
        if resource_name not in self._api_data:
            raise ResourceNotFoundError(resource_name)
        return ResourceData(resource_name, self._api_name, self._api_data[resource_name])

    def add_resource(self, resource_data):
        if not isinstance(resource_data, ResourceData):
            raise UnwrappedDataException('Resource', resource_data)
        elif resource_data.get_resource_name() in self._api_data:
            raise ResourceAlreadyExistsError(resource_data.get_resource_name())
        else:
            self._api_data.update(resource_data.to_dict())

    def update_resource(self, resource_data):
        if not isinstance(resource_data, ResourceData):
            raise UnwrappedDataException('Resource', resource_data)
        elif resource_data.get_resource_name() not in self._api_data:
            raise ResourceNotFoundError(resource_data.get_resource_name())
        else:
            self._api_data.update(resource_data.to_dict())

    def remove_resource(self, resource_name, must_exist=True):
        if must_exist and resource_name not in self._api_data:
            raise ResourceNotFoundError(resource_name)
        del self._api_data[resource_name]

    def prune(self):
        for resource_data in iter(self):
            resource_data.prune()