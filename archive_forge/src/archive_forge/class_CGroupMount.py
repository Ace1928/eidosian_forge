from __future__ import annotations
import dataclasses
import enum
import json
import os
import pathlib
import pwd
import typing as t
from ..io import (
from ..util import (
from ..config import (
from ..docker_util import (
from ..host_configs import (
from ..cgroup import (
@dataclasses.dataclass(frozen=True)
class CGroupMount:
    """Details on a cgroup mount point that is expected to be present in the container."""
    path: str
    type: t.Optional[str]
    writable: t.Optional[bool]
    state: t.Optional[CGroupState]

    def __post_init__(self):
        assert pathlib.PurePosixPath(self.path).is_relative_to(CGroupPath.ROOT)
        if self.type is None:
            assert self.state is None
        elif self.type == MountType.TMPFS:
            assert self.writable is True
            assert self.state is None
        else:
            assert self.type in (MountType.CGROUP_V1, MountType.CGROUP_V2)
            assert self.state is not None