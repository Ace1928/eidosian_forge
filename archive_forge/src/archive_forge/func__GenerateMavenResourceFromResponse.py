from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import json
from apitools.base.protorpclite import protojson
from googlecloudsdk.command_lib.artifacts import containeranalysis_util as ca_util
from googlecloudsdk.core import resources
def _GenerateMavenResourceFromResponse(response):
    """Convert Versions Describe Response to maven artifact resource name."""
    r = resources.REGISTRY.ParseRelativeName(response['name'], 'artifactregistry.projects.locations.repositories.packages.versions')
    registry = resources.REGISTRY.Clone()
    registry.RegisterApiByName('artifactregistry', 'v1')
    maven_artifacts_id = r.packagesId + ':' + r.versionsId
    maven_resource = resources.Resource.RelativeName(registry.Create('artifactregistry.projects.locations.repositories.mavenArtifacts', projectsId=r.projectsId, locationsId=r.locationsId, repositoriesId=r.repositoriesId, mavenArtifactsId=maven_artifacts_id))
    return (r.projectsId, maven_resource)