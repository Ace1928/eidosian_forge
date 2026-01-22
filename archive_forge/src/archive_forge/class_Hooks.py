from typing import Dict, List, Tuple
from . import errors, registry
from .lazy_import import lazy_import
from breezy import (
from breezy.i18n import gettext
class Hooks(dict):
    """A dictionary mapping hook name to a list of callables.

    e.g. ['FOO'] Is the list of items to be called when the
    FOO hook is triggered.
    """

    def __init__(self, module=None, member_name=None):
        """Create a new hooks dictionary.

        :param module: The module from which this hooks dictionary should be loaded
            (used for lazy hooks)
        :param member_name: Name under which this hooks dictionary should be loaded.
            (used for lazy hooks)
        """
        dict.__init__(self)
        self._callable_names = {}
        self._lazy_callable_names = {}
        self._module = module
        self._member_name = member_name

    def add_hook(self, name, doc, introduced, deprecated=None):
        """Add a hook point to this dictionary.

        :param name: The name of the hook, for clients to use when registering.
        :param doc: The docs for the hook.
        :param introduced: When the hook was introduced (e.g. (0, 15)).
        :param deprecated: When the hook was deprecated, None for
            not-deprecated.
        """
        if name in self:
            raise errors.DuplicateKey(name)
        if self._module:
            callbacks = _lazy_hooks.setdefault((self._module, self._member_name, name), [])
        else:
            callbacks = None
        hookpoint = HookPoint(name=name, doc=doc, introduced=introduced, deprecated=deprecated, callbacks=callbacks)
        self[name] = hookpoint

    def docs(self):
        """Generate the documentation for this Hooks instance.

        This introspects all the individual hooks and returns their docs as well.
        """
        hook_names = sorted(self.keys())
        hook_docs = []
        name = self.__class__.__name__
        hook_docs.append(name)
        hook_docs.append('-' * len(name))
        hook_docs.append('')
        for hook_name in hook_names:
            hook = self[hook_name]
            hook_docs.append(hook.docs())
        return '\n'.join(hook_docs)

    def get_hook_name(self, a_callable):
        """Get the name for a_callable for UI display.

        If no name has been registered, the string 'No hook name' is returned.
        We use a fixed string rather than repr or the callables module because
        the code names are rarely meaningful for end users and this is not
        intended for debugging.
        """
        name = self._callable_names.get(a_callable, None)
        if name is None and a_callable is not None:
            name = self._lazy_callable_names.get((a_callable.__module__, a_callable.__name__), None)
        if name is None:
            return 'No hook name'
        return name

    def install_named_hook_lazy(self, hook_name, callable_module, callable_member, name):
        """Install a_callable in to the hook hook_name lazily, and label it.

        :param hook_name: A hook name. See the __init__ method for the complete
            list of hooks.
        :param callable_module: Name of the module in which the callable is
            present.
        :param callable_member: Member name of the callable.
        :param name: A name to associate the callable with, to show users what
            is running.
        """
        try:
            hook = self[hook_name]
        except KeyError:
            raise UnknownHook(self.__class__.__name__, hook_name)
        try:
            hook_lazy = getattr(hook, 'hook_lazy')
        except AttributeError:
            raise errors.UnsupportedOperation(self.install_named_hook_lazy, self)
        else:
            hook_lazy(callable_module, callable_member, name)
        if name is not None:
            self.name_hook_lazy(callable_module, callable_member, name)

    def install_named_hook(self, hook_name, a_callable, name):
        """Install a_callable in to the hook hook_name, and label it name.

        :param hook_name: A hook name. See the __init__ method for the complete
            list of hooks.
        :param a_callable: The callable to be invoked when the hook triggers.
            The exact signature will depend on the hook - see the __init__
            method for details on each hook.
        :param name: A name to associate a_callable with, to show users what is
            running.
        """
        try:
            hook = self[hook_name]
        except KeyError:
            raise UnknownHook(self.__class__.__name__, hook_name)
        try:
            hook.append(a_callable)
        except AttributeError:
            hook.hook(a_callable, name)
        if name is not None:
            self.name_hook(a_callable, name)

    def uninstall_named_hook(self, hook_name, label):
        """Uninstall named hooks.

        :param hook_name: Hook point name
        :param label: Label of the callable to uninstall
        """
        try:
            hook = self[hook_name]
        except KeyError:
            raise UnknownHook(self.__class__.__name__, hook_name)
        try:
            uninstall = getattr(hook, 'uninstall')
        except AttributeError:
            raise errors.UnsupportedOperation(self.uninstall_named_hook, self)
        else:
            uninstall(label)

    def name_hook(self, a_callable, name):
        """Associate name with a_callable to show users what is running."""
        self._callable_names[a_callable] = name

    def name_hook_lazy(self, callable_module, callable_member, callable_name):
        self._lazy_callable_names[callable_module, callable_member] = callable_name