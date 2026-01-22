from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.util import apis
def TagClient():
    """Returns a client instance of the CRM Tags service."""
    return apis.GetClientInstance('cloudresourcemanager', TAGS_API_VERSION)