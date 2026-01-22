from decorator import decorator
from scipy import sparse
import importlib
import numbers
import numpy as np
import pandas as pd
import re
import warnings
def _version_check(version, min_version=None):
    if min_version is None:
        return True
    min_version = str(min_version)
    min_version_split = re.split('[^0-9]+', min_version)
    version_split = re.split('[^0-9]+', version)
    version_major = int(version_split[0])
    min_major = int(min_version_split[0])
    if min_major > version_major:
        return False
    elif min_major < version_major:
        return True
    elif len(min_version_split) == 1:
        return True
    else:
        version_minor = int(version_split[1])
        min_minor = int(min_version_split[1])
        if min_minor > version_minor:
            return False
        else:
            return True