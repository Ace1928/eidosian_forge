import datetime
import hashlib
import json
import uuid
from openstack.cloud import meta
from openstack.orchestration.util import template_format
from openstack import utils
class FakeNovaSecgroupRule:

    def __init__(self, id, from_port=None, to_port=None, ip_protocol=None, cidr=None, parent_group_id=None):
        self.id = id
        self.from_port = from_port
        self.to_port = to_port
        self.ip_protocol = ip_protocol
        if cidr:
            self.ip_range = {'cidr': cidr}
        self.parent_group_id = parent_group_id