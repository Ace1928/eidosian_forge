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
class ResourceAlreadyExistsError(ResourceMapError):
    """Exception for when a Resource being added already exists in the ResourceMap."""

    def __init__(self, api_name):
        super(ResourceAlreadyExistsError, self).__init__('[{}] API already exists in ResourceMap.'.format(api_name))