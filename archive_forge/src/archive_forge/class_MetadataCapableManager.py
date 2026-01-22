import abc
import contextlib
import copy
import hashlib
import os
from manilaclient import api_versions
from manilaclient.common import cliutils
from manilaclient import exceptions
from manilaclient import utils
class MetadataCapableManager(ManagerWithFind, metaclass=abc.ABCMeta):
    """Provides extended behavior to objects to handle key=value metadata."""
    resource_path = None
    subresource_path = None

    def get_metadata(self, resource, subresource=None):
        """Get metadata of a resource.

        :param resource: either resource object or text with its ID.
        :param subresource: either a child resource object or text with its ID
        """
        resource = getid(resource)
        if subresource:
            subresource = getid(subresource)
            resource = f'{resource}{self.subresource_path}/{subresource}'
        return self._get(f'{self.resource_path}/{resource}/metadata', 'metadata')

    def set_metadata(self, resource, metadata, subresource=None):
        """Set or update metadata for resource.

        :param resource: either resource object or text with its ID.
        :param metadata: A dictionary of key:value pairs to be set as
            resource metadata
        :param subresource: either a child resource object or text with its ID
        """
        body = {'metadata': metadata}
        resource = getid(resource)
        if subresource:
            subresource = getid(subresource)
            resource = f'{resource}{self.subresource_path}/{subresource}'
        return self._create(f'{self.resource_path}/{resource}/metadata', body, 'metadata')

    def delete_metadata(self, resource, keys, subresource=None):
        """Delete specified keys from resource metadata.

        :param resource: either resource object or text with its ID.
        :param keys: An iterable with keys of metadata items to be deleted
        :param subresource: either a child resource object or text with its ID
        """
        resource = getid(resource)
        if subresource:
            subresource = getid(subresource)
            resource = f'{resource}{self.subresource_path}/{subresource}'
        for key in keys:
            self._delete(f'{self.resource_path}/{resource}/metadata/{key}')

    def update_all_metadata(self, resource, metadata, subresource=None):
        """Update all metadata of a resource.

        :param resource: either resource object or text with its ID.
        :param metadata: A dictionary of key:value pairs of resource metadata
            to be updated
        :param subresource: either a child resource object or text with its ID
        """
        body = {'metadata': metadata}
        resource = getid(resource)
        if subresource:
            subresource = getid(subresource)
            resource = f'{resource}{self.subresource_path}/{subresource}'
        return self._update(f'{self.resource_path}/{resource}/metadata', body)