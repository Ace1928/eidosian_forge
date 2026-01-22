from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.command_lib.artifacts import requests
from googlecloudsdk.command_lib.artifacts import util
from googlecloudsdk.core import resources
def ListGenericFiles(args):
    """Lists the Generic Files stored."""
    client = requests.GetClient()
    messages = requests.GetMessages()
    project = util.GetProject(args)
    location = util.GetLocation(args)
    repo = util.GetRepo(args)
    package = args.package
    version = args.version
    version_path = resources.Resource.RelativeName(resources.REGISTRY.Create('artifactregistry.projects.locations.repositories.packages.versions', projectsId=project, locationsId=location, repositoriesId=repo, packagesId=package, versionsId=version))
    arg_filters = 'owner="{}"'.format(version_path)
    repo_path = resources.Resource.RelativeName(resources.REGISTRY.Create('artifactregistry.projects.locations.repositories', projectsId=project, locationsId=location, repositoriesId=repo))
    files = requests.ListFiles(client, messages, repo_path, arg_filters)
    return files