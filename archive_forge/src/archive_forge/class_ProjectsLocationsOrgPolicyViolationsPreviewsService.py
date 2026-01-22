from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.policysimulator.v1beta import policysimulator_v1beta_messages as messages
class ProjectsLocationsOrgPolicyViolationsPreviewsService(base_api.BaseApiService):
    """Service class for the projects_locations_orgPolicyViolationsPreviews resource."""
    _NAME = 'projects_locations_orgPolicyViolationsPreviews'

    def __init__(self, client):
        super(PolicysimulatorV1beta.ProjectsLocationsOrgPolicyViolationsPreviewsService, self).__init__(client)
        self._upload_configs = {}