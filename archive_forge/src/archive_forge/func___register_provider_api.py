import functools
import inspect
import time
import types
from oslo_log import log
import stevedore
from keystone.common import provider_api
from keystone.i18n import _
def __register_provider_api(self):
    provider_api.ProviderAPIs._register_provider_api(name=self._provides_api, obj=self)