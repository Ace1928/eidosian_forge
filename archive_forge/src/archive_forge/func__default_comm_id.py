import uuid
from typing import Optional
from warnings import warn
import comm.base_comm
import traitlets.config
from traitlets import Bool, Bytes, Instance, Unicode, default
from ipykernel.jsonutil import json_clean
from ipykernel.kernelbase import Kernel
@default('comm_id')
def _default_comm_id(self):
    return uuid.uuid4().hex