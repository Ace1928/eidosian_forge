import functools
def _make_proxy_property(attr_name):

    def proxy_property(self):
        return getattr(self._file, attr_name)
    return property(proxy_property)