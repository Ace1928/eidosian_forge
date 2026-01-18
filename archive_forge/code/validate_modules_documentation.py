from __future__ import annotations
import collections
import contextlib
import json
import os
import tarfile
import typing as t
from . import (
from ...io import (
from ...test import (
from ...target import (
from ...util import (
from ...util_common import (
from ...ansible_util import (
from ...config import (
from ...ci import (
from ...data import (
from ...host_configs import (
from ...git import (
from ...provider.source import (
Include the previous plugin content archive in the payload.