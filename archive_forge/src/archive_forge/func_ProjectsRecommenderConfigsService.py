from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.util import apis
def ProjectsRecommenderConfigsService(api_version):
    """Returns the service class for the Project recommender configs."""
    client = RecommenderClient(api_version)
    return client.projects_locations_recommenders