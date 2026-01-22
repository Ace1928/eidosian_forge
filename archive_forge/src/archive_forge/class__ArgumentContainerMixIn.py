import argparse as orig_argparse
import warnings
from autopage import argparse
class _ArgumentContainerMixIn(object):

    def add_argument_group(self, *args, **kwargs):
        group = _ArgumentGroup(self, *args, **kwargs)
        self._action_groups.append(group)
        return group

    def add_mutually_exclusive_group(self, **kwargs):
        group = _MutuallyExclusiveGroup(self, **kwargs)
        self._mutually_exclusive_groups.append(group)
        return group

    def _handle_conflict_ignore(self, action, conflicting_actions):
        _handle_conflict_ignore(self, self._option_string_actions, action, conflicting_actions)