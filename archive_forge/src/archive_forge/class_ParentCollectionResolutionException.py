from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
import re
from googlecloudsdk.api_lib.util import apis_internal
from googlecloudsdk.api_lib.util import apis_util
from googlecloudsdk.api_lib.util import resource as resource_util
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import properties
import six
from six.moves import map  # pylint: disable=redefined-builtin
from six.moves import urllib
from six.moves import zip  # pylint: disable=redefined-builtin
import uritemplate
class ParentCollectionResolutionException(Error):
    """Exception for when the parent collection cannot be computed automatically.
  """

    def __init__(self, collection, params):
        super(ParentCollectionResolutionException, self).__init__('Could not resolve the parent collection of collection [{collection}]. No collections found with parameters [{params}]'.format(collection=collection, params=', '.join(params)))