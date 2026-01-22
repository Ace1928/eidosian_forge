from .. import errors
from .resource import Collection, Model
class PluginCollection(Collection):
    model = Plugin

    def create(self, name, plugin_data_dir, gzip=False):
        """
            Create a new plugin.

            Args:
                name (string): The name of the plugin. The ``:latest`` tag is
                    optional, and is the default if omitted.
                plugin_data_dir (string): Path to the plugin data directory.
                    Plugin data directory must contain the ``config.json``
                    manifest file and the ``rootfs`` directory.
                gzip (bool): Compress the context using gzip. Default: False

            Returns:
                (:py:class:`Plugin`): The newly created plugin.
        """
        self.client.api.create_plugin(name, plugin_data_dir, gzip)
        return self.get(name)

    def get(self, name):
        """
        Gets a plugin.

        Args:
            name (str): The name of the plugin.

        Returns:
            (:py:class:`Plugin`): The plugin.

        Raises:
            :py:class:`docker.errors.NotFound` If the plugin does not
            exist.
            :py:class:`docker.errors.APIError`
                If the server returns an error.
        """
        return self.prepare_model(self.client.api.inspect_plugin(name))

    def install(self, remote_name, local_name=None):
        """
            Pull and install a plugin.

            Args:
                remote_name (string): Remote reference for the plugin to
                    install. The ``:latest`` tag is optional, and is the
                    default if omitted.
                local_name (string): Local name for the pulled plugin.
                    The ``:latest`` tag is optional, and is the default if
                    omitted. Optional.

            Returns:
                (:py:class:`Plugin`): The installed plugin
            Raises:
                :py:class:`docker.errors.APIError`
                    If the server returns an error.
        """
        privileges = self.client.api.plugin_privileges(remote_name)
        it = self.client.api.pull_plugin(remote_name, privileges, local_name)
        for data in it:
            pass
        return self.get(local_name or remote_name)

    def list(self):
        """
        List plugins installed on the server.

        Returns:
            (list of :py:class:`Plugin`): The plugins.

        Raises:
            :py:class:`docker.errors.APIError`
                If the server returns an error.
        """
        resp = self.client.api.plugins()
        return [self.prepare_model(r) for r in resp]