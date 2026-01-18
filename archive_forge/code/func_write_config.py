from yowsup.config.manager import ConfigManager
from yowsup.config.v1.config import Config
from yowsup.axolotl.manager import AxolotlManager
from yowsup.axolotl.factory import AxolotlManagerFactory
import logging
def write_config(self, config):
    logger.debug('write_config for %s' % self._profile_name)
    self._config_manager.save(self._profile_name, config)