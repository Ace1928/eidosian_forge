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
class PrivateAttributeNotFoundError(ResourceMapError):
    """Exception for when a private attribute that doesn't exist is accessed."""

    def __init__(self, data_wrapper, attribute_name):
        super(PrivateAttributeNotFoundError, self).__init__('[{}] does not have private attribute [{}].'.format(data_wrapper, attribute_name))