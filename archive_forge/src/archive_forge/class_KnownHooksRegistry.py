from typing import Dict, List, Tuple
from . import errors, registry
from .lazy_import import lazy_import
from breezy import (
from breezy.i18n import gettext
class KnownHooksRegistry(registry.Registry[str, 'Hooks']):

    def register_lazy_hook(self, hook_module_name, hook_member_name, hook_factory_member_name):
        self.register_lazy((hook_module_name, hook_member_name), hook_module_name, hook_factory_member_name)

    def iter_parent_objects(self):
        """Yield (hook_key, (parent_object, attr)) tuples for every registered
        hook, where 'parent_object' is the object that holds the hook
        instance.

        This is useful for resetting/restoring all the hooks to a known state,
        as is done in breezy.tests.TestCase._clear_hooks.
        """
        for key in self.keys():
            yield (key, self.key_to_parent_and_attribute(key))

    def key_to_parent_and_attribute(self, key):
        """Convert a known_hooks key to a (parent_obj, attr) pair.

        :param key: A tuple (module_name, member_name) as found in the keys of
            the known_hooks registry.
        :return: The parent_object of the hook and the name of the attribute on
            that parent object where the hook is kept.
        """
        parent_mod, parent_member, attr = pyutils.calc_parent_name(*key)
        return (pyutils.get_named_object(parent_mod, parent_member), attr)