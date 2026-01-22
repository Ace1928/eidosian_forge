from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import network
@dataclass
class InsecureContentStatus:
    """
    Information about insecure content on the page.
    """
    ran_mixed_content: bool
    displayed_mixed_content: bool
    contained_mixed_form: bool
    ran_content_with_cert_errors: bool
    displayed_content_with_cert_errors: bool
    ran_insecure_content_style: SecurityState
    displayed_insecure_content_style: SecurityState

    def to_json(self):
        json = dict()
        json['ranMixedContent'] = self.ran_mixed_content
        json['displayedMixedContent'] = self.displayed_mixed_content
        json['containedMixedForm'] = self.contained_mixed_form
        json['ranContentWithCertErrors'] = self.ran_content_with_cert_errors
        json['displayedContentWithCertErrors'] = self.displayed_content_with_cert_errors
        json['ranInsecureContentStyle'] = self.ran_insecure_content_style.to_json()
        json['displayedInsecureContentStyle'] = self.displayed_insecure_content_style.to_json()
        return json

    @classmethod
    def from_json(cls, json):
        return cls(ran_mixed_content=bool(json['ranMixedContent']), displayed_mixed_content=bool(json['displayedMixedContent']), contained_mixed_form=bool(json['containedMixedForm']), ran_content_with_cert_errors=bool(json['ranContentWithCertErrors']), displayed_content_with_cert_errors=bool(json['displayedContentWithCertErrors']), ran_insecure_content_style=SecurityState.from_json(json['ranInsecureContentStyle']), displayed_insecure_content_style=SecurityState.from_json(json['displayedInsecureContentStyle']))