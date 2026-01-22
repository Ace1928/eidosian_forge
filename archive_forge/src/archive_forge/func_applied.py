import os
from ... import osutils
from . import wrapper
def applied(self):
    return wrapper.quilt_applied(self.tree)