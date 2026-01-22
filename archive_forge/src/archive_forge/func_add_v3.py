from keystoneauth1 import _utils as utils
def add_v3(self, href, **kwargs):
    """Add a v3 version to the list.

        The parameters are the same as V3Discovery.
        """
    obj = V3Discovery(href, **kwargs)
    self.add_version(obj)
    return obj