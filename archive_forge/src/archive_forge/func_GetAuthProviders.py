from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
import copy
import io
from googlecloudsdk.core import exceptions as core_exceptions
from googlecloudsdk.core import yaml
from googlecloudsdk.core.util import files
def GetAuthProviders(self, name_only=True):
    try:
        providers = self[self.AUTH_PROVIDERS_KEY]
    except KeyError:
        return None
    if not providers:
        return None
    if name_only:
        return [provider['name'] for provider in providers]
    return providers