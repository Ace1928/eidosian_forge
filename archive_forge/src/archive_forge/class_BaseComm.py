import uuid
from typing import Optional
from warnings import warn
import comm.base_comm
import traitlets.config
from traitlets import Bool, Bytes, Instance, Unicode, default
from ipykernel.jsonutil import json_clean
from ipykernel.kernelbase import Kernel
class BaseComm(comm.base_comm.BaseComm):
    """The base class for comms."""
    kernel: Optional['Kernel'] = None

    def publish_msg(self, msg_type, data=None, metadata=None, buffers=None, **keys):
        """Helper for sending a comm message on IOPub"""
        if not Kernel.initialized():
            return
        data = {} if data is None else data
        metadata = {} if metadata is None else metadata
        content = json_clean(dict(data=data, comm_id=self.comm_id, **keys))
        if self.kernel is None:
            self.kernel = Kernel.instance()
        assert self.kernel.session is not None
        self.kernel.session.send(self.kernel.iopub_socket, msg_type, content, metadata=json_clean(metadata), parent=self.kernel.get_parent(), ident=self.topic, buffers=buffers)