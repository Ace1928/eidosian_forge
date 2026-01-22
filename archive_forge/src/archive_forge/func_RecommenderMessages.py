from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.util import apis
def RecommenderMessages(api_version):
    """Returns the messages module for the Resource Settings service."""
    return apis.GetMessagesModule(RECOMMENDER_API_NAME, api_version)