from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import list_pager
from googlecloudsdk.api_lib.artifacts import exceptions as ar_exceptions
from googlecloudsdk.api_lib.cloudkms import iam as kms_iam
from googlecloudsdk.api_lib.cloudresourcemanager import projects_api
from googlecloudsdk.api_lib.iam import util as iam_api
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.command_lib.iam import iam_util
from googlecloudsdk.command_lib.projects import util as project_util
from googlecloudsdk.core import resources
def GetVersionFromTag(client, messages, tag):
    """Gets a version name by a tag name."""
    get_tag_req = messages.ArtifactregistryProjectsLocationsRepositoriesPackagesTagsGetRequest(name=tag)
    get_tag_res = client.projects_locations_repositories_packages_tags.Get(get_tag_req)
    if not get_tag_res.version or len(get_tag_res.version.split('/')) != 10:
        raise ar_exceptions.ArtifactRegistryError('Internal error. Corrupted tag: {}'.format(tag))
    return get_tag_res.version.split('/')[-1]