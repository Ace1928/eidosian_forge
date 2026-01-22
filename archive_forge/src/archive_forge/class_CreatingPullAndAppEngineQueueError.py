from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import encoding
from apitools.base.py import list_pager
from googlecloudsdk.core import exceptions
import six
class CreatingPullAndAppEngineQueueError(exceptions.InternalError):
    """Error for when attempt to create a queue as both pull and App Engine."""