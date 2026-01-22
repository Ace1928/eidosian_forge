from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import exceptions as apitools_exceptions
from apitools.base.py import list_pager
from googlecloudsdk.api_lib.services import exceptions
from googlecloudsdk.api_lib.util import apis
def _ValidateConsumer(consumer):
    for prefix in _VALID_CONSUMER_PREFIX:
        if consumer.startswith(prefix):
            return
    raise exceptions.Error('invalid consumer format "%s".' % consumer)