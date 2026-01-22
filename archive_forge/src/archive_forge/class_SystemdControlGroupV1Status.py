from __future__ import annotations
import dataclasses
import enum
import json
import os
import pathlib
import re
import socket
import time
import urllib.parse
import typing as t
from .util import (
from .util_common import (
from .config import (
from .thread import (
from .cgroup import (
class SystemdControlGroupV1Status(enum.Enum):
    """The state of the cgroup v1 systemd hierarchy on the container host."""
    SUBSYSTEM_MISSING = 'The systemd cgroup subsystem was not found.'
    FILESYSTEM_NOT_MOUNTED = 'The "/sys/fs/cgroup/systemd" filesystem is not mounted.'
    MOUNT_TYPE_NOT_CORRECT = 'The "/sys/fs/cgroup/systemd" mount type is not correct.'
    VALID = 'The "/sys/fs/cgroup/systemd" mount is valid.'