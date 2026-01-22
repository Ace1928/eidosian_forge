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
class AmbiguousAPIException(Error):
    """Exception for when two APIs try to define a resource."""

    def __init__(self, collection, base_urls):
        super(AmbiguousAPIException, self).__init__('collection [{collection}] defined in multiple APIs: {apis}'.format(collection=collection, apis=repr(base_urls)))