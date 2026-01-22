import uuid
from keystoneclient.tests.unit.v3 import utils
class EndpointTestUtils(object):
    """Mixin class with shared methods between Endpoint Filter & Policy."""

    def new_ref(self, **kwargs):
        kwargs.setdefault('id', uuid.uuid4().hex)
        kwargs.setdefault('enabled', True)
        return kwargs

    def new_endpoint_ref(self, **kwargs):
        kwargs = self.new_ref(**kwargs)
        kwargs.setdefault('interface', 'public')
        kwargs.setdefault('region', uuid.uuid4().hex)
        kwargs.setdefault('service_id', uuid.uuid4().hex)
        kwargs.setdefault('url', uuid.uuid4().hex)
        return kwargs

    def new_endpoint_group_ref(self, **kwargs):
        kwargs.setdefault('id', uuid.uuid4().hex)
        kwargs.setdefault('name', uuid.uuid4().hex)
        kwargs.setdefault('description', uuid.uuid4().hex)
        kwargs.setdefault('filters')
        return kwargs