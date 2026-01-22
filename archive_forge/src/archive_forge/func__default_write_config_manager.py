import os.path
import typing as t
from jupyter_core.paths import jupyter_config_dir, jupyter_config_path
from traitlets import Instance, List, Unicode, default, observe
from traitlets.config import LoggingConfigurable
from jupyter_server.config_manager import BaseJSONConfigManager, recursive_update
@default('write_config_manager')
def _default_write_config_manager(self):
    return BaseJSONConfigManager(config_dir=self.write_config_dir)