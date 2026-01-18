from __future__ import annotations
import datetime
import json
import sys
from typing import Any, IO, Iterable, TYPE_CHECKING
from coverage import __version__
from coverage.report_core import get_analysis_to_report
from coverage.results import Analysis, Numbers
from coverage.types import TMorf, TLineNo
Extract the relevant report data for a single file.