import copy
import json
import optparse
import os
import pickle
import sys
from urllib import parse
from troveclient.compat import client
from troveclient.compat import exceptions
def _safe_exec(self, func, *args, **kwargs):
    if not self.debug:
        try:
            return func(*args, **kwargs)
        except Exception:
            print(sys.exc_info()[1])
            return None
    else:
        return func(*args, **kwargs)