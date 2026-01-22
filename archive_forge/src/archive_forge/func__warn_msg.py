from __future__ import absolute_import
import os
from pickle import PicklingError
from ray.cloudpickle.cloudpickle import *  # noqa
from ray.cloudpickle.cloudpickle_fast import CloudPickler, dumps, dump  # noqa
def _warn_msg(obj, method, exc):
    return f"{method}({str(obj)}) failed.\nTo check which non-serializable variables are captured in scope, re-run the ray script with 'RAY_PICKLE_VERBOSE_DEBUG=1'."