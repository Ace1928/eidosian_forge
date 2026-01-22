import enum
class Rule(object):
    """Base class for conversion rules."""

    def __init__(self, module_prefix):
        self._prefix = module_prefix

    def matches(self, module_name):
        return module_name.startswith(self._prefix + '.') or module_name == self._prefix