import copy
import os
import tempfile
import fixtures
import yaml
from openstack.config import cloud_region
from openstack.tests.unit import base
def _write_yaml(obj):
    with tempfile.NamedTemporaryFile(delete=False, suffix='.yaml') as obj_yaml:
        obj_yaml.write(yaml.safe_dump(obj).encode('utf-8'))
        return obj_yaml.name