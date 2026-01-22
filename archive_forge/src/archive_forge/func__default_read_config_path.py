import os.path
import typing as t
from jupyter_core.paths import jupyter_config_dir, jupyter_config_path
from traitlets import Instance, List, Unicode, default, observe
from traitlets.config import LoggingConfigurable
from jupyter_server.config_manager import BaseJSONConfigManager, recursive_update
@default('read_config_path')
def _default_read_config_path(self):
    return [os.path.join(p, self.config_dir_name) for p in jupyter_config_path()]