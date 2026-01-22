from typing import Callable
from typing import Dict
from typing import Optional
from typing import Type
from typing import Union
from rpy2.rinterface_lib import openrlib
from rpy2.rinterface_lib import _rinterface_capi as _rinterface
class DummyMissingRpy2Map(object):

    def __init__(self, *args, **kwargs):
        raise NotImplementedError('The default object mapper class is no set.')