from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import os
from apitools.base.py import exceptions as apitools_exceptions
from googlecloudsdk.api_lib.cloudresourcemanager import projects_api
from googlecloudsdk.api_lib.compute import base_classes
from googlecloudsdk.api_lib.util import exceptions as api_lib_util_exceptions
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions as calliope_exceptions
from googlecloudsdk.command_lib.projects import util as command_lib_util
from googlecloudsdk.core import config
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core.credentials import creds as c_creds
from googlecloudsdk.core.credentials import store as c_store
def WarnIfSettingProjectWithNoAccess(scope, project):
    """Warn if setting 'core/project' config to inaccessible project."""
    if scope == properties.Scope.USER and properties.VALUES.core.account.Get():
        project_ref = command_lib_util.ParseProject(project)
        try:
            with base.WithLegacyQuota():
                projects_api.Get(project_ref, disable_api_enablement_check=True)
        except (apitools_exceptions.HttpError, c_store.NoCredentialsForAccountException, api_lib_util_exceptions.HttpException):
            log.warning('You do not appear to have access to project [{}] or it does not exist.'.format(project))
            return True
    return False