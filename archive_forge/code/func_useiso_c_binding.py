import copy
from .auxfuncs import (
from ._isocbind import isoc_kindmap
def useiso_c_binding(rout):
    useisoc = False
    for key, value in rout['vars'].items():
        kind_value = value.get('kindselector', {}).get('kind')
        if kind_value in isoc_kindmap:
            return True
    return useisoc