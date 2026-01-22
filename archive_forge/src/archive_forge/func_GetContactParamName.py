from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import list_pager
from googlecloudsdk.api_lib.util import apis
def GetContactParamName(version=DEFAULT_API_VERSION):
    return _CONTACT_TYPES_BY_VERSION[version]['param_name']