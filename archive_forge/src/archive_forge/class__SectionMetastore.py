from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import enum
import functools
import os
import re
import sys
import textwrap
from googlecloudsdk.core import argv_utils
from googlecloudsdk.core import config
from googlecloudsdk.core import exceptions
from googlecloudsdk.core.configurations import named_configs
from googlecloudsdk.core.configurations import properties_file as prop_files_lib
from googlecloudsdk.core.docker import constants as const_lib
from googlecloudsdk.core.resource import resource_printer_types as formats
from googlecloudsdk.core.util import encoding
from googlecloudsdk.core.util import http_proxy_types
from googlecloudsdk.core.util import scaled_integer
from googlecloudsdk.generated_clients.apis import apis_map
import six
class _SectionMetastore(_Section):
    """Contains the properties for the 'metastore' section."""

    class Tier(enum.Enum):
        developer = 1
        enterprise = 3

    def TierValidator(self, tier):
        if tier is None:
            return
        if tier not in [x.name for x in list(_SectionMetastore.Tier)]:
            raise InvalidValueError('tier `{0}` must be one of: [developer, enterprise]'.format(tier))

    def __init__(self):
        super(_SectionMetastore, self).__init__('metastore')
        self.location = self._Add('location', help_text='Default location to use when working with Dataproc Metastore. When a `location` is required but not provided by a flag, the command will fall back to this value, if set.')
        self.tier = self._Add('tier', validator=self.TierValidator, help_text=textwrap.dedent('        Default tier to use when creating Dataproc Metastore services.\n        When a `tier` is required but not provided by a flag,\n        the command will fall back to this value, if set.\n        +\n        Valid values are:\n            *   `developer` - The developer tier provides limited scalability\n            and no fault tolerance. Good for low-cost proof-of-concept.\n            *   `enterprise` - The enterprise tier provides multi-zone high\n            availability, and sufficient scalability for enterprise-level\n            Dataproc Metastore workloads.'), choices=[x.name for x in list(_SectionMetastore.Tier)])