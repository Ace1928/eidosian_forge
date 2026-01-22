from __future__ import annotations
import os
import tempfile
import uuid
import typing as t
import urllib.parse
from ..encoding import (
from ..config import (
from ..git import (
from ..http import (
from ..util import (
from . import (
Return the last successful commit from git history that is found in the given commit list, or None.