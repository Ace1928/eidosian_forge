from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import debugger
from . import emulation
from . import io
from . import page
from . import runtime
from . import security
@event_class('Network.reportingApiReportAdded')
@dataclass
class ReportingApiReportAdded:
    """
    **EXPERIMENTAL**

    Is sent whenever a new report is added.
    And after 'enableReportingApi' for all existing reports.
    """
    report: ReportingApiReport

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> ReportingApiReportAdded:
        return cls(report=ReportingApiReport.from_json(json['report']))