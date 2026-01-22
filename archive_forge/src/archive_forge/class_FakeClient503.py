import testtools
from heatclient import exc
from heatclient.v1 import services
class FakeClient503(object):

    def get(self, *args, **kwargs):
        assert args[0] == '/services'
        raise exc.HTTPServiceUnavailable()