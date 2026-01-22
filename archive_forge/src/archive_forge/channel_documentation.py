from __future__ import absolute_import
import datetime
import uuid
from googleapiclient import errors
from googleapiclient import _helpers as util
import six
Update a channel with information from the response of watch().

    When a request is sent to watch() a resource, the response returned
    from the watch() request is a dictionary with updated channel information,
    such as the resource_id, which is needed when stopping a subscription.

    Args:
      resp: dict, The response from a watch() method.
    