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
class UnwrappedDataException(ResourceMapError):
    """Exception for when unwrapped data is added to the map."""

    def __init__(self, field_type, data):
        super(UnwrappedDataException, self).__init__('The following data must be wrapped in a(n) {}Data wrapper prior to being added to the resource map: [{}]'.format(field_type, data))