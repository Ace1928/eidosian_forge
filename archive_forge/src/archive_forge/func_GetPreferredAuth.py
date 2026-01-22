from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
import copy
import io
from googlecloudsdk.core import exceptions as core_exceptions
from googlecloudsdk.core import yaml
from googlecloudsdk.core.util import files
def GetPreferredAuth(self):
    if self.version == AUTH_VERSION_2_ALPHA:
        return self[self.PREFERRED_AUTH_KEY]
    else:
        raise YamlConfigObjectFieldError(self.PREFERRED_AUTH_KEY, self.__class__.__name__, 'requires config version [{}]'.format(AUTH_VERSION_2_ALPHA))