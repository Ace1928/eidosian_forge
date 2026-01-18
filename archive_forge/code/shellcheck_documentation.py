from __future__ import annotations
import os
import typing as t
from xml.etree.ElementTree import (
from . import (
from ...test import (
from ...target import (
from ...util import (
from ...util_common import (
from ...config import (
Return the given list of test targets, filtered to include only those relevant for the test.