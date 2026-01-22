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
def ConvertGCRImageString(image_string):
    """Converts GCR image string to AR format. Leaves non-GCR strings as-is."""
    location_map = {'us.gcr.io': 'us', 'gcr.io': 'us', 'eu.gcr.io': 'europe', 'asia.gcr.io': 'asia'}
    matches = re.match(GCR_DOCKER_REPO_REGEX, image_string)
    if matches:
        return ('{}-docker.pkg.dev/{}/{}/{}'.format(location_map[matches.group('repo')], matches.group('project'), matches.group('repo'), matches.group('image')), matches.group('project'), True)
    matches = re.match(GCR_DOCKER_DOMAIN_SCOPED_REPO_REGEX, image_string)
    if matches:
        return ('{}-docker.pkg.dev/{}/{}/{}'.format(location_map[matches.group('repo')], matches.group('project'), matches.group('repo'), matches.group('image')), matches.group('project'), True)
    return (image_string, None, False)