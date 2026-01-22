import re
from . import errors
def _compile_and_collapse(self):
    """Actually compile the requested regex"""
    self._real_regex = self._real_re_compile(*self._regex_args, **self._regex_kwargs)
    for attr in self._regex_attributes_to_copy:
        setattr(self, attr, getattr(self._real_regex, attr))