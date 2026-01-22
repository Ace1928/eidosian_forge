import abc
from typing import Any
class DebuggerEventHandler(_with_metaclass(abc.ABCMeta)):
    """
    Implement this to receive lifecycle events from the debugger
    """

    def on_debugger_modules_loaded(self, **kwargs):
        """
        This method invoked after all debugger modules are loaded. Useful for importing and/or patching debugger
        modules at a safe time
        :param kwargs: This is intended to be flexible dict passed from the debugger.
        Currently passes the debugger version
        """