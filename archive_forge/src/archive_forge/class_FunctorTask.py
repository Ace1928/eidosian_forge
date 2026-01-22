import abc
import copy
import functools
from oslo_utils import reflection
from taskflow import atom
from taskflow import logging
from taskflow.types import notifier
from taskflow.utils import misc
class FunctorTask(Task):
    """Adaptor to make a task from a callable.

    Take any callable pair and make a task from it.

    NOTE(harlowja): If a name is not provided the function/method name of
    the ``execute`` callable will be used as the name instead (the name of
    the ``revert`` callable is not used).
    """

    def __init__(self, execute, name=None, provides=None, requires=None, auto_extract=True, rebind=None, revert=None, version=None, inject=None):
        if not callable(execute):
            raise ValueError('Function to use for executing must be callable')
        if revert is not None:
            if not callable(revert):
                raise ValueError('Function to use for reverting must be callable')
        if name is None:
            name = reflection.get_callable_name(execute)
        super(FunctorTask, self).__init__(name, provides=provides, inject=inject)
        self._execute = execute
        self._revert = revert
        if version is not None:
            self.version = version
        mapping = self._build_arg_mapping(execute, requires, rebind, auto_extract)
        self.rebind, exec_requires, self.optional = mapping
        if revert:
            revert_mapping = self._build_arg_mapping(revert, requires, rebind, auto_extract)
        else:
            revert_mapping = (self.rebind, exec_requires, self.optional)
        self.revert_rebind, revert_requires, self.revert_optional = revert_mapping
        self.requires = exec_requires.union(revert_requires)

    def execute(self, *args, **kwargs):
        return self._execute(*args, **kwargs)

    def revert(self, *args, **kwargs):
        if self._revert:
            return self._revert(*args, **kwargs)
        else:
            return None