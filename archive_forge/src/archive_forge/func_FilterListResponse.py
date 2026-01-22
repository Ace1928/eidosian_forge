from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.core import log
from six.moves import filter  # pylint: disable=redefined-builtin
def FilterListResponse(response, unused_args):
    return list(filter(_IsPublicVersion, response))