from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.sourcerepo.v1 import sourcerepo_v1_messages as messages
Updates the Cloud Source Repositories configuration of the project.

      Args:
        request: (SourcerepoProjectsUpdateConfigRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ProjectConfig) The response message.
      