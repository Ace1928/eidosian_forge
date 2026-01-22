from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from dateutil import tz
from googlecloudsdk.core.util import times
def SnakeCaseToCamelCase(name):
    words = name.split('_')
    return words[0].lower() + ''.join([w[0].upper() + w[1:].lower() for w in words[1:]])