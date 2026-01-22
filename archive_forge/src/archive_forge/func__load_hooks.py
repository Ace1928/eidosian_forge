import abc
import inspect
from stevedore import extension
from cliff import _argparse
def _load_hooks(self):
    if self.app and self.cmd_name:
        namespace = '{}.{}'.format(self.app.command_manager.namespace, self.cmd_name.replace(' ', '_'))
        self._hooks = extension.ExtensionManager(namespace=namespace, invoke_on_load=True, invoke_kwds={'command': self})
    else:
        self._hooks = []
    return