from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import browser
from . import network
from . import page
@dataclass
class SharedStorageAccessParams:
    """
    Bundles the parameters for shared storage access events whose
    presence/absence can vary according to SharedStorageAccessType.
    """
    script_source_url: typing.Optional[str] = None
    operation_name: typing.Optional[str] = None
    serialized_data: typing.Optional[str] = None
    urls_with_metadata: typing.Optional[typing.List[SharedStorageUrlWithMetadata]] = None
    key: typing.Optional[str] = None
    value: typing.Optional[str] = None
    ignore_if_present: typing.Optional[bool] = None

    def to_json(self):
        json = dict()
        if self.script_source_url is not None:
            json['scriptSourceUrl'] = self.script_source_url
        if self.operation_name is not None:
            json['operationName'] = self.operation_name
        if self.serialized_data is not None:
            json['serializedData'] = self.serialized_data
        if self.urls_with_metadata is not None:
            json['urlsWithMetadata'] = [i.to_json() for i in self.urls_with_metadata]
        if self.key is not None:
            json['key'] = self.key
        if self.value is not None:
            json['value'] = self.value
        if self.ignore_if_present is not None:
            json['ignoreIfPresent'] = self.ignore_if_present
        return json

    @classmethod
    def from_json(cls, json):
        return cls(script_source_url=str(json['scriptSourceUrl']) if 'scriptSourceUrl' in json else None, operation_name=str(json['operationName']) if 'operationName' in json else None, serialized_data=str(json['serializedData']) if 'serializedData' in json else None, urls_with_metadata=[SharedStorageUrlWithMetadata.from_json(i) for i in json['urlsWithMetadata']] if 'urlsWithMetadata' in json else None, key=str(json['key']) if 'key' in json else None, value=str(json['value']) if 'value' in json else None, ignore_if_present=bool(json['ignoreIfPresent']) if 'ignoreIfPresent' in json else None)