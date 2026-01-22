import copy
import functools
import queue
import warnings
import dogpile.cache
import keystoneauth1.exceptions
import keystoneauth1.session
import requests.models
import requestsexceptions
from openstack import _log
from openstack.cloud import _object_store
from openstack.cloud import _utils
from openstack.cloud import meta
import openstack.config
from openstack.config import cloud_region as cloud_region_mod
from openstack import exceptions
from openstack import proxy
from openstack import utils
def _make_cache_key(self, namespace, fn):
    fname = fn.__name__
    if namespace is None:
        name_key = self.name
    else:
        name_key = '%s:%s' % (self.name, namespace)

    def generate_key(*args, **kwargs):
        arg_key = ''
        kw_keys = sorted(kwargs.keys())
        kwargs_key = ','.join(['%s:%s' % (k, kwargs[k]) for k in kw_keys if k != 'cache'])
        ans = '_'.join([str(name_key), fname, arg_key, kwargs_key])
        return ans
    return generate_key