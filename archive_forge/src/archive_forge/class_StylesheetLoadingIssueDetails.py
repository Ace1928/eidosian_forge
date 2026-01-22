from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import dom
from . import network
from . import page
from . import runtime
@dataclass
class StylesheetLoadingIssueDetails:
    """
    This issue warns when a referenced stylesheet couldn't be loaded.
    """
    source_code_location: SourceCodeLocation
    style_sheet_loading_issue_reason: StyleSheetLoadingIssueReason
    failed_request_info: typing.Optional[FailedRequestInfo] = None

    def to_json(self):
        json = dict()
        json['sourceCodeLocation'] = self.source_code_location.to_json()
        json['styleSheetLoadingIssueReason'] = self.style_sheet_loading_issue_reason.to_json()
        if self.failed_request_info is not None:
            json['failedRequestInfo'] = self.failed_request_info.to_json()
        return json

    @classmethod
    def from_json(cls, json):
        return cls(source_code_location=SourceCodeLocation.from_json(json['sourceCodeLocation']), style_sheet_loading_issue_reason=StyleSheetLoadingIssueReason.from_json(json['styleSheetLoadingIssueReason']), failed_request_info=FailedRequestInfo.from_json(json['failedRequestInfo']) if 'failedRequestInfo' in json else None)