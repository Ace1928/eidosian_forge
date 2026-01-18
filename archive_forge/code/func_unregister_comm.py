from __future__ import annotations
import contextlib
import logging
import typing as t
import uuid
from traitlets.utils.importstring import import_item
import comm
def unregister_comm(self, comm: BaseComm) -> None:
    """Unregister a comm, and close its counterpart"""
    comm = self.comms.pop(comm.comm_id)