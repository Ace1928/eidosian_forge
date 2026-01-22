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
class RequiredFieldOmittedException(UserError):
    """A command line that was given did not specify a field."""

    def __init__(self, collection_name, expected):
        super(RequiredFieldOmittedException, self).__init__('value for field [{expected}] in collection [{collection_name}] is required but was not provided'.format(expected=expected, collection_name=collection_name))