from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
from concurrent import futures
import encodings.idna  # pylint: disable=unused-import
import json
import mimetypes
import os
import re
from apitools.base.py import encoding
from apitools.base.py import exceptions as apitools_exceptions
from containerregistry.client import docker_name
from containerregistry.client.v2_2 import docker_http
from containerregistry.client.v2_2 import docker_image
from googlecloudsdk.api_lib import artifacts
from googlecloudsdk.api_lib.artifacts import exceptions as ar_exceptions
from googlecloudsdk.api_lib.container.images import util
from googlecloudsdk.api_lib.util import common_args
from googlecloudsdk.api_lib.util import waiter
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.artifacts import remote_repo_util
from googlecloudsdk.command_lib.artifacts import requests as ar_requests
from googlecloudsdk.command_lib.artifacts import upgrade_util
from googlecloudsdk.command_lib.projects import util as project_util
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
from googlecloudsdk.core import yaml
from googlecloudsdk.core.console import console_attr
from googlecloudsdk.core.console import console_io
from googlecloudsdk.core.console import progress_tracker
from googlecloudsdk.core.resource import resource_printer
from googlecloudsdk.core.util import edit
from googlecloudsdk.core.util import parallel
import requests
def GetRedirectionEnablementReport(project):
    """Prints a redirection enablement report and returns mis-configured repos.

  This checks all the GCR repositories in the supplied project and checks if
  they each have a repository in Artifact Registry create to be the redirection
  target. It prints a report as it validates.

  Args:
    project: The project to validate

  Returns:
    A list of the GCR repos that do not have a redirection repo configured in
    Artifact Registry.
  """
    missing_repos = []
    repo_report = []
    con = console_attr.GetConsoleAttr()
    location = getattr(properties.VALUES.artifacts, 'location').Get()
    for gcr_repo in gcr_repos:
        if gcr_base != 'gcr.io' and location and (location != gcr_repo['location']):
            continue
        report_line = [gcr_repo['host'], gcr_repo['location']]
        ar_repo_name = 'projects/{}/locations/{}/repositories/{}'.format(project, gcr_repo['location'], gcr_repo['repository'])
        try:
            ar_repo = ar_requests.GetRepository(ar_repo_name)
            report_line.append(con.Colorize(ar_repo.name, 'green'))
        except apitools_exceptions.HttpNotFoundError:
            report_line.append(con.Colorize('None Found. Can create repo named "{}"'.format(gcr_repo['repository']), 'yellow'))
            missing_repos.append(gcr_repo)
        repo_report.append(report_line)
    log.status.Print(f'Repository report for {project}:\n')
    printer = resource_printer.Printer('table', out=log.status)
    printer.AddHeading([con.Emphasize('Container Registry Host', bold=True), con.Emphasize('Location', bold=True), con.Emphasize('Artifact Registry Repository', bold=True)])
    for line in repo_report:
        printer.AddRecord(line)
    printer.Finish()
    log.status.Print()
    return missing_repos