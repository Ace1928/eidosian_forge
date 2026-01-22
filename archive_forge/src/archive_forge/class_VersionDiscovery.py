from keystoneauth1 import _utils as utils
class VersionDiscovery(DiscoveryBase):
    """A Version element for non-keystone services without microversions.

    Provides some default values and helper methods for creating a microversion
    endpoint version structure. Clients should use this instead of creating
    their own structures.

    :param string href: The url that this entry should point to.
    :param string id: The version id that should be reported.
    """

    def __init__(self, href, id, **kwargs):
        super(VersionDiscovery, self).__init__(id, **kwargs)
        self.add_link(href)