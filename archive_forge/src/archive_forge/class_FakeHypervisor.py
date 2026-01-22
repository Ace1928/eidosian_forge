import datetime
import hashlib
import json
import uuid
from openstack.cloud import meta
from openstack.orchestration.util import template_format
from openstack import utils
class FakeHypervisor:

    def __init__(self, id, hostname):
        self.id = id
        self.hypervisor_hostname = hostname