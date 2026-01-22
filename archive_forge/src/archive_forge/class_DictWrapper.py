import copy
from collections.abc import Mapping
class DictWrapper(dict):
    """
    Wrap accesses to a dictionary so that certain values (those starting with
    the specified prefix) are passed through a function before being returned.
    The prefix is removed before looking up the real value.

    Used by the SQL construction code to ensure that values are correctly
    quoted before being used.
    """

    def __init__(self, data, func, prefix):
        super().__init__(data)
        self.func = func
        self.prefix = prefix

    def __getitem__(self, key):
        """
        Retrieve the real value after stripping the prefix string (if
        present). If the prefix is present, pass the value through self.func
        before returning, otherwise return the raw value.
        """
        use_func = key.startswith(self.prefix)
        key = key.removeprefix(self.prefix)
        value = super().__getitem__(key)
        if use_func:
            return self.func(value)
        return value