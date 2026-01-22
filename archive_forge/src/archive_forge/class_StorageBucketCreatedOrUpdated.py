from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import browser
from . import network
from . import page
@event_class('Storage.storageBucketCreatedOrUpdated')
@dataclass
class StorageBucketCreatedOrUpdated:
    bucket_info: StorageBucketInfo

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> StorageBucketCreatedOrUpdated:
        return cls(bucket_info=StorageBucketInfo.from_json(json['bucketInfo']))