from __future__ import annotations
import io
import os
import typing
from pathlib import Path
from ._types import (
from ._utils import (

        Return the length of the multipart encoded content, or `None` if
        any of the files have a length that cannot be determined upfront.
        