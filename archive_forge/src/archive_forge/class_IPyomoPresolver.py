from pyomo.common.plugin_base import (
class IPyomoPresolver(Interface):

    def get_actions(self):
        """Return a list of presolve actions, in the order in which
        they will be applied."""

    def activate_action(self, action):
        """Activate an action, but leave its default rank"""

    def deactivate_action(self, action):
        """Deactivate an action"""

    def set_actions(self, actions):
        """Set presolve action list"""

    def presolve(self, instance):
        """Apply the presolve actions to this instance, and return the
        revised instance"""