from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.core import exceptions
class InvalidConfigYaml(exceptions.Error):
    """For when a membership configuration is invalid or could not be parsed."""