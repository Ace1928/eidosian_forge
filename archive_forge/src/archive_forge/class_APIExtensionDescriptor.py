import abc
from neutron_lib._i18n import _
from neutron_lib import constants
class APIExtensionDescriptor(ExtensionDescriptor):
    """Base class that defines the contract for extensions.

    Concrete implementations of API extensions should first provide
    an API definition in neutron_lib.api.definitions. The API
    definition module (object reference) can then be specified as a
    class level attribute on the concrete extension.

    For example::

        from neutron_lib.api.definitions import provider_net
        from neutron_lib.api import extensions


        class Providernet(extensions.APIExtensionDescriptor):
            api_definition = provider_net
            # nothing else needed if default behavior is acceptable


    If extension implementations need to override the default behavior of
    this class they can override the respective method directly.
    """
    api_definition = _UNSET

    @classmethod
    def _assert_api_definition(cls, attr=None):
        if cls.api_definition == _UNSET:
            raise NotImplementedError(_('Extension module API definition not set.'))
        if attr and getattr(cls.api_definition, attr, _UNSET) == _UNSET:
            raise NotImplementedError(_("Extension module API definition does not define '%s'") % attr)

    @classmethod
    def get_name(cls):
        """The name of the API definition."""
        cls._assert_api_definition('NAME')
        return cls.api_definition.NAME

    @classmethod
    def get_alias(cls):
        """The alias for the API definition."""
        cls._assert_api_definition('ALIAS')
        return cls.api_definition.ALIAS

    @classmethod
    def get_description(cls):
        """Friendly description for the API definition."""
        cls._assert_api_definition('DESCRIPTION')
        return cls.api_definition.DESCRIPTION

    @classmethod
    def get_updated(cls):
        """The timestamp when the API definition was last updated."""
        cls._assert_api_definition('UPDATED_TIMESTAMP')
        return cls.api_definition.UPDATED_TIMESTAMP

    @classmethod
    def get_extended_resources(cls, version):
        """Retrieve the extended resource map for the API definition.

        :param version: The API version to retrieve the resource attribute
            map for.
        :returns: The extended resource map for the underlying API definition
            if the version is 2.0. The extended resource map returned includes
            both the API definition's RESOURCE_ATTRIBUTE_MAP and
            SUB_RESOURCE_ATTRIBUTE_MAP where applicable. If the version is
            not 2.0, an empty dict is returned.
        """
        if version == '2.0':
            cls._assert_api_definition('RESOURCE_ATTRIBUTE_MAP')
            cls._assert_api_definition('SUB_RESOURCE_ATTRIBUTE_MAP')
            sub_attrs = cls.api_definition.SUB_RESOURCE_ATTRIBUTE_MAP or {}
            return dict(list(cls.api_definition.RESOURCE_ATTRIBUTE_MAP.items()) + list(sub_attrs.items()))
        else:
            return {}

    @classmethod
    def get_required_extensions(cls):
        """Returns the API definition's required extensions."""
        cls._assert_api_definition('REQUIRED_EXTENSIONS')
        return cls.api_definition.REQUIRED_EXTENSIONS

    @classmethod
    def get_optional_extensions(cls):
        """Returns the API definition's optional extensions."""
        cls._assert_api_definition('OPTIONAL_EXTENSIONS')
        return cls.api_definition.OPTIONAL_EXTENSIONS

    @classmethod
    def update_attributes_map(cls, extended_attributes, extension_attrs_map=None):
        """Update attributes map for this extension.

        Behaves like ExtensionDescriptor.update_attributes_map(), but
        if extension_attrs_map is not given the dict returned from
        self.get_extended_resources('2.0') is used.
        """
        if extension_attrs_map is None:
            extension_attrs_map = cls.get_extended_resources('2.0')
        super().update_attributes_map(extended_attributes, extension_attrs_map=extension_attrs_map)