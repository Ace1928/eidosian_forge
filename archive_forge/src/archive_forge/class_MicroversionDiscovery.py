from keystoneauth1 import _utils as utils
class MicroversionDiscovery(DiscoveryBase):
    """A Version element that has microversions.

    Provides some default values and helper methods for creating a microversion
    endpoint version structure. Clients should use this instead of creating
    their own structures.

    :param string href: The url that this entry should point to.
    :param string id: The version id that should be reported.
    :param string min_version: The minimum supported microversion. (optional)
    :param string max_version: The maximum supported microversion. (optional)
    """

    def __init__(self, href, id, min_version='', max_version='', **kwargs):
        super(MicroversionDiscovery, self).__init__(id, **kwargs)
        self.add_link(href)
        self.min_version = min_version
        self.max_version = max_version

    @property
    def min_version(self):
        return self.get('min_version')

    @min_version.setter
    def min_version(self, value):
        self['min_version'] = value

    @property
    def max_version(self):
        return self.get('max_version')

    @max_version.setter
    def max_version(self, value):
        self['max_version'] = value