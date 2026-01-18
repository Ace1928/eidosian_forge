from __future__ import print_function
import sys
import os
import types
import traceback
from abc import abstractmethod
def version_tnf(t1, t2=None):
    from ruamel.yaml import version_info
    if version_info < t1:
        return True
    if t2 is not None and version_info < t2:
        return None
    return False