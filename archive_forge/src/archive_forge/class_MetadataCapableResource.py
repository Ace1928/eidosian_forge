import abc
import contextlib
import copy
import hashlib
import os
from manilaclient import api_versions
from manilaclient.common import cliutils
from manilaclient import exceptions
from manilaclient import utils
class MetadataCapableResource(Resource, metaclass=abc.ABCMeta):
    superresource = None

    def _get_subresource_and_resource(self, superresource):
        resource = self
        subresource = None
        superresource = superresource or self.superresource
        if superresource is not None:
            resource = superresource
            subresource = self
        return (resource, subresource)

    def get_metadata(self, superresource=None):
        """Get metadata of a resource

        :param superresource: either a parent resource object or text with
            its ID. Required for sub-resources such as share export
            locations which do not include a reference to the parent object
            by default
        """
        resource, subresource = self._get_subresource_and_resource(superresource)
        return self.manager.get_metadata(resource, subresource=subresource)

    def set_metadata(self, metadata, superresource=None):
        """Set or update metadata for the resource.

        :param metadata: A dictionary of key:value pairs to be set as
            resource metadata
        :param superresource: either a parent resource object or text with
            its ID. Required for sub-resources such as share share export
            locations which do not include a reference to the parent object
            by default
        """
        resource, subresource = self._get_subresource_and_resource(superresource)
        return self.manager.set_metadata(resource, metadata, subresource=subresource)

    def delete_metadata(self, keys, superresource=None):
        """Delete specified keys from the given resource.

        :param keys: An iterable with keys of metadata items to be deleted
        :param superresource: either a parent resource object or text with
            its ID. Required for sub-resources such as share share export
            locations which do not include a reference to the parent object
            by default
        """
        resource, subresource = self._get_subresource_and_resource(superresource)
        return self.manager.delete_metadata(resource, keys, subresource=subresource)

    def update_all_metadata(self, metadata, superresource=None):
        """Update all metadata for this resource.

        :param metadata: A dictionary of key:value pairs of resource metadata
            to be updated
        :param superresource: either a parent resource object or text with
            its ID. Required for sub-resources such as share share export
            locations which do not include a reference to the parent object
            by default
        """
        resource, subresource = self._get_subresource_and_resource(superresource)
        return self.manager.update_all_metadata(resource, metadata, subresource=subresource)