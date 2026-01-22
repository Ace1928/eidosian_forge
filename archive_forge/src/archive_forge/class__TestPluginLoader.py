import uuid
import fixtures
from keystoneauth1 import discover
from keystoneauth1 import loading
from keystoneauth1 import plugin
class _TestPluginLoader(loading.BaseLoader):

    def __init__(self, plugin):
        super(_TestPluginLoader, self).__init__()
        self._plugin = plugin

    def create_plugin(self, **kwargs):
        return self._plugin

    def get_options(self):
        return []