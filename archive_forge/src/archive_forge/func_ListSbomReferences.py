from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import hashlib
import json
import random
import re
from apitools.base.py import encoding
from apitools.base.py import exceptions as apitools_exceptions
from containerregistry.client import docker_creds
from containerregistry.client import docker_name
from containerregistry.client.v2_2 import docker_http as v2_2_docker_http
from containerregistry.client.v2_2 import docker_image as v2_2_image
from containerregistry.client.v2_2 import docker_image_list as v2_2_image_list
from googlecloudsdk.api_lib.artifacts import exceptions as ar_exceptions
from googlecloudsdk.api_lib.cloudkms import base as cloudkms_base
from googlecloudsdk.api_lib.container.images import util as gcr_util
from googlecloudsdk.api_lib.containeranalysis import filter_util
from googlecloudsdk.api_lib.containeranalysis import requests as ca_requests
from googlecloudsdk.api_lib.storage import storage_api
from googlecloudsdk.api_lib.storage import storage_util
from googlecloudsdk.command_lib.artifacts import docker_util
from googlecloudsdk.command_lib.artifacts import requests as ar_requests
from googlecloudsdk.command_lib.artifacts import util
from googlecloudsdk.command_lib.projects import util as project_util
from googlecloudsdk.core import log
from googlecloudsdk.core import resources
from googlecloudsdk.core import transports
from googlecloudsdk.core.util import files
import requests
import six
from six.moves import urllib
def ListSbomReferences(args):
    """Lists SBOM references in a given project.

  Args:
    args: User input arguments.

  Returns:
    List of SBOM references.
  """
    resource = args.resource
    prefix = args.resource_prefix
    dependency = args.dependency
    if resource and (prefix or dependency) or (prefix and dependency):
        raise ar_exceptions.InvalidInputValueError('Cannot specify more than one of the flags --dependency, --resource and --resource-prefix.')
    filters = filter_util.ContainerAnalysisFilter().WithKinds(['SBOM_REFERENCE'])
    project = util.GetProject(args)
    if dependency:
        dependency_filters = filter_util.ContainerAnalysisFilter().WithKinds(['PACKAGE']).WithCustomFilter('noteProjectId="goog-analysis" AND dependencyPackageName="{}"'.format(dependency))
        package_occs = list(ca_requests.ListOccurrences(project, dependency_filters.GetFilter(), None))
        if not package_occs:
            return []
        images = set((_EnsurePrefix(o.resourceUri, 'https://') for o in package_occs))
        filters.WithResources(images)
    if resource:
        resource_uri = _RemovePrefix(resource, 'https://')
        resource_uris = ['https://{}'.format(resource_uri), resource_uri]
        try:
            artifact = ProcessArtifact(resource_uri)
            if resource_uri != artifact.resource_uri:
                resource_uris = resource_uris + ['https://{}'.format(artifact.resource_uri), artifact.resource_uri]
            if artifact.project:
                project = artifact.project
        except (ar_exceptions.InvalidInputValueError, docker_name.BadNameException):
            log.status.Print('Failed to resolve the artifact. Filter on the URI directly.')
            pass
        filters.WithResources(resource_uris)
    if prefix:
        path_prefix = _RemovePrefix(prefix, 'https://')
        filters.WithResourcePrefixes(['https://{}'.format(path_prefix), path_prefix])
    if dependency:
        occs = ca_requests.ListOccurrencesWithFilters(project, filters.GetChunkifiedFilters())
    else:
        occs = ca_requests.ListOccurrences(project, filters.GetFilter(), args.page_size)
    if resource:
        return _VerifyGCSObjects(occs)
    return [SbomReference(occ, {}) for occ in occs]