import uuid
from keystoneclient import exceptions
from keystoneclient.tests.unit.v3 import utils
from keystoneclient.v3 import roles
from testtools import matchers
def _new_domain_ref(self, **kwargs):
    kwargs.setdefault('enabled', True)
    kwargs.setdefault('name', uuid.uuid4().hex)
    return kwargs