import datetime
import hashlib
import json
import uuid
from openstack.cloud import meta
from openstack.orchestration.util import template_format
from openstack import utils
class FakeRecordset:

    def __init__(self, zone, id, name, type_, description, ttl, records):
        self.zone = zone
        self.id = id
        self.name = name
        self.type_ = type_
        self.description = description
        self.ttl = ttl
        self.records = records