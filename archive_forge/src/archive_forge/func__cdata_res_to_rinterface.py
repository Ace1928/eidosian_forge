from typing import Callable
from typing import Dict
from typing import Optional
from typing import Type
from typing import Union
from rpy2.rinterface_lib import openrlib
from rpy2.rinterface_lib import _rinterface_capi as _rinterface
def _cdata_res_to_rinterface(function):

    def _(*args, **kwargs):
        cdata = function(*args, **kwargs)
        return _cdata_to_rinterface(cdata)
    return _