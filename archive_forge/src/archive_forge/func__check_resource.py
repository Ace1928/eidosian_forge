import functools
import typing as ty
import urllib
from urllib.parse import urlparse
import iso8601
import jmespath
from keystoneauth1 import adapter
from openstack import _log
from openstack import exceptions
from openstack import resource
def _check_resource(strict=False):

    def wrap(method):

        def check(self, expected, actual=None, *args, **kwargs):
            if strict and actual is not None and (not isinstance(actual, resource.Resource)):
                raise ValueError('A %s must be passed' % expected.__name__)
            elif isinstance(actual, resource.Resource) and (not isinstance(actual, expected)):
                raise ValueError('Expected %s but received %s' % (expected.__name__, actual.__class__.__name__))
            return method(self, expected, actual, *args, **kwargs)
        return check
    return wrap