import os
import pathlib
import uuid
from typing import Any, Dict, List, NewType, Optional, Union, cast
from dataclasses import dataclass, fields
from jupyter_core.utils import ensure_async
from tornado import web
from traitlets import Instance, TraitError, Unicode, validate
from traitlets.config.configurable import LoggingConfigurable
from jupyter_server.traittypes import InstanceFromClasses
@dataclass
class KernelSessionRecord:
    """A record object for tracking a Jupyter Server Kernel Session.

    Two records that share a session_id must also share a kernel_id, while
    kernels can have multiple session (and thereby) session_ids
    associated with them.
    """
    session_id: Optional[str] = None
    kernel_id: Optional[str] = None

    def __eq__(self, other: object) -> bool:
        """Whether a record equals another."""
        if isinstance(other, KernelSessionRecord):
            condition1 = self.kernel_id and self.kernel_id == other.kernel_id
            condition2 = all([self.session_id == other.session_id, self.kernel_id is None or other.kernel_id is None])
            if any([condition1, condition2]):
                return True
            if all([self.session_id, self.session_id == other.session_id, self.kernel_id != other.kernel_id]):
                msg = 'A single session_id can only have one kernel_id associated with. These two KernelSessionRecords share the same session_id but have different kernel_ids. This should not be possible and is likely an issue with the session records.'
                raise KernelSessionRecordConflict(msg)
        return False

    def update(self, other: 'KernelSessionRecord') -> None:
        """Updates in-place a kernel from other (only accepts positive updates"""
        if not isinstance(other, KernelSessionRecord):
            msg = "'other' must be an instance of KernelSessionRecord."
            raise TypeError(msg)
        if other.kernel_id and self.kernel_id and (other.kernel_id != self.kernel_id):
            msg = "Could not update the record from 'other' because the two records conflict."
            raise KernelSessionRecordConflict(msg)
        for field in fields(self):
            if hasattr(other, field.name) and getattr(other, field.name):
                setattr(self, field.name, getattr(other, field.name))