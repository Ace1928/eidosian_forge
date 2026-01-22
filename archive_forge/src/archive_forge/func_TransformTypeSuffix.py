from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.compute import constants
from googlecloudsdk.api_lib.compute import instance_utils
from googlecloudsdk.api_lib.compute import path_simplifier
from googlecloudsdk.core.resource import resource_transform
import six
def TransformTypeSuffix(uri, undefined=''):
    """Get the type and the name of the object the uri refers to."""
    try:
        return '/'.join(uri.split('/')[-2:]) or undefined
    except (AttributeError, IndexError, TypeError):
        pass
    return undefined