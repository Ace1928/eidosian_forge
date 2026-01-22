from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import encoding
from googlecloudsdk.api_lib.dataproc import util
Get a field name by suffix and truncate it.

    The one_of fields in server response have their type name as field key.
    One can retrieve the name of those fields by iterating through all the
    fields.

    Args:
      suffix: the suffix to match.

    Returns:
      The first matched truncated field name.

    Raises:
      AttributeError: Error occur when there is no match for the suffix.

    Usage Example:
      In server response:
      {
        ...
        "sparkJob":{
          ...
        }
        ...
      }
      type = helper.getTruncatedFieldNameBySuffix('Job')
    