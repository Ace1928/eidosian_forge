import json
from troveclient import base
from troveclient import common
class ConfigurationParameters(base.ManagerWithFind):
    """Manage :class:`ConfigurationParameters` information."""
    resource_class = ConfigurationParameter

    def parameters(self, datastore, version):
        """Get a list of valid parameters that can be changed."""
        return self._list('/datastores/%s/versions/%s/parameters' % (datastore, version), 'configuration-parameters')

    def get_parameter(self, datastore, version, key):
        """Get a list of valid parameters that can be changed."""
        return self._get('/datastores/%s/versions/%s/parameters/%s' % (datastore, version, key))

    def parameters_by_version(self, version):
        """Get a list of valid parameters that can be changed."""
        return self._list('/datastores/versions/%s/parameters' % version, 'configuration-parameters')

    def get_parameter_by_version(self, version, key):
        """Get a list of valid parameters that can be changed."""
        return self._get('/datastores/versions/%s/parameters/%s' % (version, key))

    def list(self):
        pass