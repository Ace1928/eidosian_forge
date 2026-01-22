from __future__ import absolute_import, division, print_function
import os
def _default_haver(key):
    try:
        self.get_option(key)
        return True
    except KeyError:
        return False