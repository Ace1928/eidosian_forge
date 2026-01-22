import uuid
from typing import Optional
from warnings import warn
import comm.base_comm
import traitlets.config
from traitlets import Bool, Bytes, Instance, Unicode, default
from ipykernel.jsonutil import json_clean
from ipykernel.kernelbase import Kernel
@default('kernel')
def _default_kernel(self):
    if Kernel.initialized():
        return Kernel.instance()
    return None