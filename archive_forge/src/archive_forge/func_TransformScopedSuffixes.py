from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.compute import constants
from googlecloudsdk.api_lib.compute import instance_utils
from googlecloudsdk.api_lib.compute import path_simplifier
from googlecloudsdk.core.resource import resource_transform
import six
def TransformScopedSuffixes(uris, undefined=''):
    """Get just the scoped part of the object the uri refers to."""
    if uris:
        try:
            return sorted([path_simplifier.ScopedSuffix(uri) for uri in uris])
        except TypeError:
            pass
    return undefined