from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
import json
from apitools.base.py import encoding
from googlecloudsdk.command_lib.storage.resources import full_resource_formatter
from googlecloudsdk.command_lib.storage.resources import resource_reference
from googlecloudsdk.command_lib.storage.resources import resource_util
class GcsAnywhereCacheResource(resource_reference.CloudResource):
    """Holds Anywhere Cache metadata."""

    def __init__(self, admission_policy=None, anywhere_cache_id=None, bucket=None, create_time=None, id_string=None, kind=None, metadata=None, pending_update=None, state=None, storage_url=None, ttl=None, update_time=None, zone=None):
        self.admission_policy = admission_policy
        self.anywhere_cache_id = anywhere_cache_id
        self.bucket = bucket
        self.create_time = create_time
        self.id = id_string
        self.kind = kind
        self.metadata = metadata
        self.pending_update = pending_update
        self.state = state
        self.storage_url = storage_url
        self.ttl = ttl
        self.update_time = update_time
        self.zone = zone

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return NotImplemented
        return self.admission_policy == other.admission_policy and self.anywhere_cache_id == other.anywhere_cache_id and (self.bucket == other.bucket) and (self.create_time == other.create_time) and (self.id == other.id) and (self.kind == other.kind) and (self.metadata == other.metadata) and (self.pending_update == other.pending_update) and (self.state == other.state) and (self.storage_url == other.storage_url) and (self.ttl == other.ttl) and (self.update_time == other.update_time) and (self.zone == other.zone)