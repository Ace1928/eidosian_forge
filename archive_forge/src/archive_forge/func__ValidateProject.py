from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import re
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.core import properties
def _ValidateProject(flag_value):
    if not re.match('^[a-z0-9-]+$', flag_value):
        raise exceptions.InvalidArgumentException('project', flag_value)