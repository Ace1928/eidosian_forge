from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
from googlecloudsdk.command_lib.util.resource_map.declarative import declarative_map
from googlecloudsdk.core import exceptions
class ResourceNameTranslator(object):
    """Name translation utility to convert between resource identifier types."""

    class ResourceNameTranslatorWrapper(object):
        """Data wrapper for resource objects returned by get_resource()."""

        def __init__(self, resource):
            self._resource = resource

        @property
        def asset_inventory_type(self):
            return self._resource.asset_inventory_type

        @property
        def krm_kind(self):
            return KrmKind(self._resource.krm_group, self._resource.krm_kind)

        @property
        def collection_name(self):
            return self._resource.get_full_collection_name()

        @property
        def resource_data(self):
            return self._resource
    _translator_instance = None

    def __new__(cls):
        if not cls._translator_instance:
            cls._translator_instance = super(ResourceNameTranslator, cls).__new__(cls)
            cls._translator_instance.populate_name_mappings(declarative_map.DeclarativeMap())
        return cls._translator_instance

    def __iter__(self):
        for resource in self.collection_map.values():
            yield resource

    def populate_name_mappings(self, resource_map):
        """Populates name maps for constant time access to resources."""
        self.ai_map = {}
        self.krm_map = {}
        self.collection_map = {}
        for api in resource_map:
            for resource in api:
                wrapped_resource = self.ResourceNameTranslatorWrapper(resource)
                if resource.has_metadata_field('asset_inventory_type'):
                    self.ai_map[resource.asset_inventory_type] = wrapped_resource
                    self.krm_map[KrmKind(resource.krm_group, resource.krm_kind)] = wrapped_resource
                    self.collection_map[resource.get_full_collection_name()] = wrapped_resource

    def find_krmkinds_by_kind(self, kind):
        """Gets a list of KrmKind keys based on krm kind values."""
        return [x for x in self.krm_map.keys() if x.krm_kind == kind]

    def get_resource(self, asset_inventory_type=None, krm_kind=None, collection_name=None):
        """Gets resource object wrapper given resource identifier."""
        _validate_translator_args(asset_inventory_type=asset_inventory_type, krm_kind=krm_kind, collection_name=collection_name)
        if asset_inventory_type:
            if not self.is_translatable(asset_inventory_type=asset_inventory_type):
                raise ResourceIdentifierNotFoundError(asset_inventory_type)
            return self.ai_map[asset_inventory_type]
        if krm_kind:
            if not isinstance(krm_kind, tuple):
                raise ResourceNameTranslatorError('[krm_kind] must be of type [tuple(krm_group, krm_type)]')
            elif not self.is_translatable(krm_kind=krm_kind):
                raise ResourceIdentifierNotFoundError(krm_kind)
            return self.krm_map[KrmKind(*krm_kind)]
        if collection_name:
            if not self.is_translatable(collection_name=collection_name):
                raise ResourceIdentifierNotFoundError(collection_name)
            return self.collection_map[collection_name]

    def is_translatable(self, asset_inventory_type=None, krm_kind=None, collection_name=None):
        """Returns true if given asset identifier exists in translation maps."""
        if asset_inventory_type:
            return asset_inventory_type in self.ai_map
        elif krm_kind:
            return KrmKind(*krm_kind) in self.krm_map
        elif collection_name:
            return collection_name in self.collection_map
        else:
            return False