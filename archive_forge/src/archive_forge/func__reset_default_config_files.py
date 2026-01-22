import fixtures
from oslo_config import cfg
def _reset_default_config_files(self):
    if not hasattr(self.conf, 'default_config_files'):
        return
    if self._default_config_files:
        self.conf.default_config_files = self._default_config_files
    else:
        self.conf.default_config_files = None