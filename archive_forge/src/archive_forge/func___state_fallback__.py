from __future__ import absolute_import, division, print_function
def __state_fallback__(self):
    raise ValueError('Cannot find method: {0}'.format(self._method(self._state())))