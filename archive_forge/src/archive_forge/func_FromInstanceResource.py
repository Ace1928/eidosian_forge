from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import re
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core.console import console_io
from six.moves import filter  # pylint: disable=redefined-builtin
from six.moves import map  # pylint: disable=redefined-builtin
@classmethod
def FromInstanceResource(cls, instance):
    match = re.match(cls._INSTANCE_NAME_PATTERN, instance.name)
    service = match.group('service')
    version = match.group('version')
    return cls(service, version, instance.id, instance)