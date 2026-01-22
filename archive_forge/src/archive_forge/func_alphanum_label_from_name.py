import re
from pyomo.common.deprecation import deprecated
from pyomo.core.base.componentuid import ComponentUID
def alphanum_label_from_name(name):
    if name is None:
        raise RuntimeError('Illegal name=None supplied to alphanum_label_from_name function')
    return str.translate(name, _alphanum_translation_table)