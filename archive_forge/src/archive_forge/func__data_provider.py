import threading
from werkzeug import wrappers
from tensorboard import errors
from tensorboard import plugin_util
from tensorboard.plugins import base_plugin
from tensorboard.plugins.debugger_v2 import debug_data_provider
from tensorboard.backend import http_util
@property
def _data_provider(self):
    if self._underlying_data_provider is not None:
        return self._underlying_data_provider
    with self._data_provider_init_lock:
        if self._underlying_data_provider is not None:
            return self._underlying_data_provider
        dp = debug_data_provider.LocalDebuggerV2DataProvider(self._logdir)
        self._underlying_data_provider = dp
        return dp