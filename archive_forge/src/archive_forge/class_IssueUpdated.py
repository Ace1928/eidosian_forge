from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
@event_class('Cast.issueUpdated')
@dataclass
class IssueUpdated:
    """
    This is fired whenever the outstanding issue/error message changes.
    ``issueMessage`` is empty if there is no issue.
    """
    issue_message: str

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> IssueUpdated:
        return cls(issue_message=str(json['issueMessage']))