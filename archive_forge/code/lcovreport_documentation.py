from __future__ import annotations
import base64
import hashlib
import sys
from typing import IO, Iterable, TYPE_CHECKING
from coverage.plugin import FileReporter
from coverage.report_core import get_analysis_to_report
from coverage.results import Analysis, Numbers
from coverage.types import TMorf
Produces the lcov data for a single file.

        This currently supports both line and branch coverage,
        however function coverage is not supported.
        