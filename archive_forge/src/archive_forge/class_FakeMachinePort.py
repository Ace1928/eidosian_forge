import datetime
import hashlib
import json
import uuid
from openstack.cloud import meta
from openstack.orchestration.util import template_format
from openstack import utils
class FakeMachinePort:

    def __init__(self, id, address, node_id):
        self.uuid = id
        self.address = address
        self.node_uuid = node_id