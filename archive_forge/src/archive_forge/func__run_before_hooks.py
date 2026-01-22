import abc
import inspect
from stevedore import extension
from cliff import _argparse
def _run_before_hooks(self, parsed_args):
    """Calls before() method of the hooks.

        This method is intended to be called from the run() method before
        take_action() is called.

        This method should only be overridden by developers creating new
        command base classes and only if it is necessary to have different
        hook processing behavior.
        """
    for hook in self._hooks:
        ret = hook.obj.before(parsed_args)
        if ret is not None:
            parsed_args = ret
    return parsed_args