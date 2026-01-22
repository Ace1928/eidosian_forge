import copy
import abc
import logging
import six
def _get_state_names(self, state):
    if isinstance(state, (list, tuple, set)):
        for res in state:
            for inner in self._get_state_names(res):
                yield inner
    else:
        yield (self.machine.state_cls.separator.join(self.machine._get_enum_path(state)) if hasattr(state, 'name') else state)