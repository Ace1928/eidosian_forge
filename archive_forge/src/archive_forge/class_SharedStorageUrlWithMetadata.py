from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import browser
from . import network
from . import page
@dataclass
class SharedStorageUrlWithMetadata:
    """
    Bundles a candidate URL with its reporting metadata.
    """
    url: str
    reporting_metadata: typing.List[SharedStorageReportingMetadata]

    def to_json(self):
        json = dict()
        json['url'] = self.url
        json['reportingMetadata'] = [i.to_json() for i in self.reporting_metadata]
        return json

    @classmethod
    def from_json(cls, json):
        return cls(url=str(json['url']), reporting_metadata=[SharedStorageReportingMetadata.from_json(i) for i in json['reportingMetadata']])