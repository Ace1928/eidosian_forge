from boto.beanstalk.layer1 import Layer1
import boto.beanstalk.response
from boto.exception import BotoServerError
import boto.beanstalk.exception as exception
class Layer1Wrapper(object):

    def __init__(self, *args, **kwargs):
        self.api = Layer1(*args, **kwargs)

    def __getattr__(self, name):
        try:
            return beanstalk_wrapper(getattr(self.api, name), name)
        except AttributeError:
            raise AttributeError('%s has no attribute %r' % (self, name))