import warnings
from .._ossighelper import wakeup_on_signal, register_sigint_fallback
from ..overrides import override, deprecated_init, wrap_list_store_sort_func
from ..module import get_introspection_module
from gi import PyGIWarning
from gi.repository import GLib
import sys
def add_action_entries(self, entries, user_data=None):
    """
        The add_action_entries() method is a convenience function for creating
        multiple Gio.SimpleAction instances and adding them to a Gio.ActionMap.
        Each action is constructed as per one entry.

        :param list entries:
            List of entry tuples for add_action() method. The entry tuple can
            vary in size with the following information:

                * The name of the action. Must be specified.
                * The callback to connect to the "activate" signal of the
                  action. Since GLib 2.40, this can be None for stateful
                  actions, in which case the default handler is used. For
                  boolean-stated actions with no parameter, this is a toggle.
                  For other state types (and parameter type equal to the state
                  type) this will be a function that just calls change_state
                  (which you should provide).
                * The type of the parameter that must be passed to the activate
                  function for this action, given as a single GLib.Variant type
                  string (or None for no parameter)
                * The initial state for this action, given in GLib.Variant text
                  format. The state is parsed with no extra type information, so
                  type tags must be added to the string if they are necessary.
                  Stateless actions should give None here.
                * The callback to connect to the "change-state" signal of the
                  action. All stateful actions should provide a handler here;
                  stateless actions should not.

        :param user_data:
            The user data for signal connections, or None
        """
    try:
        iter(entries)
    except TypeError:
        raise TypeError('entries must be iterable')

    def _process_action(name, activate=None, parameter_type=None, state=None, change_state=None):
        if parameter_type:
            if not GLib.VariantType.string_is_valid(parameter_type):
                raise TypeError("The type string '%s' given as the parameter type for action '%s' is not a valid GVariant type string. " % (parameter_type, name))
            variant_parameter = GLib.VariantType.new(parameter_type)
        else:
            variant_parameter = None
        if state is not None:
            variant_state = GLib.Variant.parse(None, state, None, None)
            action = Gio.SimpleAction.new_stateful(name, variant_parameter, variant_state)
            if change_state is not None:
                action.connect('change-state', change_state, user_data)
        else:
            if change_state is not None:
                raise ValueError("Stateless action '%s' should give None for 'change_state', not '%s'." % (name, change_state))
            action = Gio.SimpleAction(name=name, parameter_type=variant_parameter)
        if activate is not None:
            action.connect('activate', activate, user_data)
        self.add_action(action)
    for entry in entries:
        _process_action(*entry)