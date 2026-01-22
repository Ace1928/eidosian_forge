from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import re
from apitools.base.py import exceptions as api_exceptions
from googlecloudsdk.api_lib.artifacts import exceptions as ar_exceptions
from googlecloudsdk.api_lib.util import common_args
from googlecloudsdk.api_lib.util import waiter
from googlecloudsdk.command_lib.artifacts import containeranalysis_util as ca_util
from googlecloudsdk.command_lib.artifacts import requests as ar_requests
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
from googlecloudsdk.core.console import console_io
def GetDockerImages(resource, args):
    """Gets Docker images."""
    limit = args.limit
    if args.filter is not None:
        limit = None
    order_by = common_args.ParseSortByArg(args.sort_by)
    if order_by is not None:
        if ',' in order_by:
            order_by = None
            limit = None
    if isinstance(resource, DockerRepo):
        _ValidateDockerRepo(resource.GetRepositoryName())
        log.status.Print('Listing items under project {}, location {}, repository {}.\n'.format(resource.project, resource.location, resource.repo))
        return _GetDockerPackagesAndVersions(resource, args.include_tags, args.page_size, order_by, limit)
    elif isinstance(resource, DockerImage):
        _ValidateDockerRepo(resource.docker_repo.GetRepositoryName())
        log.status.Print('Listing items under project {}, location {}, repository {}.\n'.format(resource.docker_repo.project, resource.docker_repo.location, resource.docker_repo.repo))
        return _GetDockerVersions(resource, args.include_tags, args.page_size, order_by, limit, search_subdirs=True)
    return []