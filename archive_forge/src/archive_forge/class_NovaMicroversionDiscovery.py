from keystoneauth1 import _utils as utils
class NovaMicroversionDiscovery(DiscoveryBase):
    """A Version element with nova-style microversions.

    Provides some default values and helper methods for creating a microversion
    endpoint version structure. Clients should use this instead of creating
    their own structures.

    :param href: The url that this entry should point to.
    :param string id: The version id that should be reported.
    :param string min_version: The minimum microversion supported. (optional)
    :param string version: The maximum microversion supported. (optional)
    """

    def __init__(self, href, id, min_version=None, version=None, **kwargs):
        super(NovaMicroversionDiscovery, self).__init__(id, **kwargs)
        self.add_link(href)
        self.min_version = min_version
        self.version = version

    @property
    def min_version(self):
        return self.get('min_version')

    @min_version.setter
    def min_version(self, value):
        if value:
            self['min_version'] = value

    @property
    def version(self):
        return self.get('version')

    @version.setter
    def version(self, value):
        if value:
            self['version'] = value