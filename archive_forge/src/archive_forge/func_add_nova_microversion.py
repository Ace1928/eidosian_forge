from keystoneauth1 import _utils as utils
def add_nova_microversion(self, href, id, **kwargs):
    """Add a nova microversion version to the list.

        The parameters are the same as NovaMicroversionDiscovery.
        """
    obj = NovaMicroversionDiscovery(href=href, id=id, **kwargs)
    self.add_version(obj)
    return obj