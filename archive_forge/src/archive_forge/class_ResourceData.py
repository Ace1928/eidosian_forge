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
class ResourceData(object):
    """Data wrapper for a Resource object in the ResourceMap metadata file.

  Attributes:
    _resource_name: Name of the resource.
    _api_name: Name of the parent api.
    _resource_data: Metadata for the resource.
  """

    def __init__(self, resource_name, api_name, resource_data):
        self._resource_name = resource_name
        self._api_name = api_name
        self._resource_data = resource_data

    def __getattr__(self, metadata_field):
        """Returns metadata value or TrackLevelResourceData object.

    Attribute being accessed will be either a metadata field for the resource,
    or the release track (GA, BETA, or ALPHA). If the attribute is a metadata
    field the appropriate value will be returned from self._resource_data. If
    the atatribute is a release track, a TrackLevelResourceData object will be
    returned. This enables both of the following usecases:

      `value = map.api.resource.metadata_field` OR
      'value = map.api.resource.ALPHA.metadata_field`

    Args:
      metadata_field: Field or release track being accessed

    Returns:
      Metadata field value OR TrackLevelResourceData object.

    Raises:
      MetadataNotFoundError: Metadata field does not exist.
      PrivateAttributeNotFoundError: Private attribute doesn't exist in object.

    """
        if metadata_field in _RELEASE_TRACKS:
            return self.get_release_track_data(metadata_field)
        elif metadata_field.startswith('_'):
            raise PrivateAttributeNotFoundError('ResourceData', metadata_field)
        else:
            return self.get_metadata(metadata_field)

    def __setattr__(self, metadata_field, value):
        """Sets the specified metadata field to the provided value.

    If the object is not yet instantiated, then standard __setattr__ behavior
    is observed, allowing for proper object instantiation. After initialization,
    the specified metadata field within self._resource_data is set to the
    provided value

    Args:
      metadata_field: Metadata field to set the value for.
      value: Value to set the specified metadata field to.

    Returns:
      True
    """
        if metadata_field.startswith('_'):
            super(ResourceData, self).__setattr__(metadata_field, value)
        elif metadata_field not in self._resource_data:
            self.add_metadata(metadata_field, value)
        else:
            self.update_metadata(metadata_field, value)

    def __eq__(self, other):
        return self.to_dict() == other.to_dict()

    def __contains__(self, metadata_field):
        return self.has_metadata_field(metadata_field)

    def prune(self):
        """Removes any redundant metadata specifications between track and top."""
        for track in _RELEASE_TRACKS:
            if track in self._resource_data:
                track_resource_data = self._resource_data[track]
                for key in list(track_resource_data.keys()):
                    if key in self._resource_data and self._resource_data[key] == track_resource_data[key]:
                        del track_resource_data[key]
                if not track_resource_data:
                    del self._resource_data[track]

    def to_dict(self):
        return {self.get_resource_name(): self._resource_data}

    def has_metadata_field(self, metadata_field):
        return metadata_field in self._resource_data

    def get_resource_name(self):
        return self._resource_name

    def get_api_name(self):
        return self._api_name

    def get_full_collection_name(self):
        return '{}.{}'.format(self.get_api_name(), self.get_resource_name())

    def get_metadata(self, metadata_field):
        if metadata_field not in self._resource_data:
            raise MetadataNotFoundError(self._resource_name, metadata_field)
        return self._resource_data[metadata_field]

    def get_release_track_data(self, release_track):
        return TrackLevelResourceData(self._resource_name, self._api_name, self._resource_data, track=release_track)

    def add_metadata(self, metadata_field, value):
        if metadata_field in self._resource_data:
            raise MetadataAlreadyExistsError(self._resource_name, metadata_field)
        else:
            self._resource_data[metadata_field] = value

    def update_metadata(self, metadata_field, value):
        if metadata_field not in self._resource_data:
            raise MetadataNotFoundError(self._resource_name, metadata_field)
        else:
            self._resource_data[metadata_field] = value

    def remove_metadata(self, metadata_field):
        if metadata_field not in self._resource_data:
            raise MetadataNotFoundError(self._resource_name, metadata_field)
        else:
            del self._resource_data[metadata_field]