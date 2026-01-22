import os
import oslo_config.cfg
from oslo_config import sources
class EnvironmentConfigurationSource(sources.ConfigurationSource):
    """A configuration source for options in the environment."""

    @staticmethod
    def get_name(group_name, option_name):
        """Return the expected environment variable name for the given option.

        :param group_name: The group name or None. Defaults to 'DEFAULT' if
            None.
        :param option_name: The option name.
        :returns: Th expected environment variable name.
        """
        group_name = group_name or 'DEFAULT'
        return 'OS_{}__{}'.format(group_name.upper(), option_name.upper())

    def get(self, group_name, option_name, opt):
        env_name = self.get_name(group_name, option_name)
        try:
            value = os.environ[env_name]
            loc = oslo_config.cfg.LocationInfo(oslo_config.cfg.Locations.environment, env_name)
            return (value, loc)
        except KeyError:
            return (sources._NoValue, None)