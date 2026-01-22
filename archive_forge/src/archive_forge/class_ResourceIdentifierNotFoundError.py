from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
from googlecloudsdk.command_lib.util.resource_map.declarative import declarative_map
from googlecloudsdk.core import exceptions
class ResourceIdentifierNotFoundError(ResourceNameTranslatorError):
    """Exception for when a resource is not translatable."""

    def __init__(self, resource_identifier):
        super(ResourceIdentifierNotFoundError, self).__init__('Unable to locate resource via identifier: [{}].'.format(resource_identifier))