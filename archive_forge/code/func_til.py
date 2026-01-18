import atexit
import inspect
import os
import pprint
import re
import subprocess
import textwrap
def til(tnames):
    tnames = self.feature_implies_c(tnames)
    tnames = self.feature_sorted(tnames, reverse=True)
    for i, n in enumerate(tnames):
        if not self.feature_supported[n].get(keyisfalse, True):
            tnames = tnames[:i + 1]
            break
    return tnames