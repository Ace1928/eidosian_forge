from __future__ import absolute_import
from __future__ import unicode_literals
import json
import os
import bootstrapping
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.core import config
from googlecloudsdk.core import context_aware
from googlecloudsdk.core import log
from googlecloudsdk.core import metrics
from googlecloudsdk.core import properties
from googlecloudsdk.core.credentials import gce as c_gce
from googlecloudsdk.core.util import encoding
from googlecloudsdk.core.util import files
def _MaybeAddBotoOption(args, section, name, value):
    if value is None:
        return
    args.append('-o')
    args.append('{section}:{name}={value}'.format(section=section, name=name, value=value))