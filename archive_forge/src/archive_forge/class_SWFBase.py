import time
from functools import wraps
from boto.swf.layer1 import Layer1
from boto.swf.layer1_decisions import Layer1Decisions
class SWFBase(object):
    name = None
    domain = None
    aws_access_key_id = None
    aws_secret_access_key = None
    region = None

    def __init__(self, **kwargs):
        for credkey in ('aws_access_key_id', 'aws_secret_access_key'):
            if DEFAULT_CREDENTIALS.get(credkey):
                setattr(self, credkey, DEFAULT_CREDENTIALS[credkey])
        for kwarg in kwargs:
            setattr(self, kwarg, kwargs[kwarg])
        self._swf = Layer1(self.aws_access_key_id, self.aws_secret_access_key, region=self.region)

    def __repr__(self):
        rep_str = str(self.name)
        if hasattr(self, 'version'):
            rep_str += '-' + str(getattr(self, 'version'))
        return '<%s %r at 0x%x>' % (self.__class__.__name__, rep_str, id(self))