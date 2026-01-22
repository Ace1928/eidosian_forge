from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import abc
from googlecloudsdk.calliope import base as calliope_base
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import yaml
from googlecloudsdk.core import yaml_validator
from googlecloudsdk.core.util import files
import six
class ResourceMapInitializationError(ResourceMapError):
    """Exception for when an error occurs while initializing the resource map."""

    def __init__(self, init_error):
        super(ResourceMapInitializationError, self).__init__('Error while initializing resource map: [{}]'.format(init_error))