from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.core import resources
def GetClientAlphaInstance():
    return apis.GetClientInstance(DEFAULT_API_NAME, ALPHA_API_VERSION)