import abc
from neutron_lib.api.definitions import portbindings
class _TypeDriverBase(object, metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def get_type(self):
        """Get driver's network type.

        :returns: network_type value handled by this driver
        """

    @abc.abstractmethod
    def initialize(self):
        """Perform driver initialization.

        Called after all drivers have been loaded and the database has
        been initialized. No abstract methods defined below will be
        called prior to this method being called.
        """

    @abc.abstractmethod
    def is_partial_segment(self, segment):
        """Return True if segment is a partially specified segment.

        :param segment: segment dictionary
        :returns: boolean
        """

    @abc.abstractmethod
    def validate_provider_segment(self, segment):
        """Validate attributes of a provider network segment.

        :param segment: segment dictionary using keys defined above
        :raises: neutron_lib.exceptions.InvalidInput if invalid

        Called outside transaction context to validate the provider
        attributes for a provider network segment. Raise InvalidInput
        if:

         - any required attribute is missing
         - any prohibited or unrecognized attribute is present
         - any attribute value is not valid

        The network_type attribute is present in segment, but
        need not be validated.
        """

    @abc.abstractmethod
    def get_mtu(self, physical):
        """Get driver's network MTU.

        :returns: mtu maximum transmission unit

        Returns the mtu for the network based on the config values and
        the network type.
        """