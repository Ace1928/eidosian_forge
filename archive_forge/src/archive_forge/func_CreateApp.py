from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import exceptions as apitools_exceptions
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core.console import console_io
def CreateApp(api_client, project, region, suppress_warning=False, service_account=None):
    """Create an App Engine app in the given region.

  Prints info about the app being created and displays a progress tracker.

  Args:
    api_client: The App Engine Admin API client
    project: The GCP project
    region: The region to create the app
    suppress_warning: True if user doesn't need to be warned this is
        irreversible.
    service_account: The app level service account for the App Engine app.

  Raises:
    AppAlreadyExistsError if app already exists
  """
    if not suppress_warning:
        log.status.Print('You are creating an app for project [{project}].'.format(project=project))
        if service_account:
            log.status.Print('Designating app-level default service account to be [{service_account}].'.format(service_account=service_account))
        log.warning(APP_CREATE_WARNING)
    try:
        api_client.CreateApp(region, service_account=service_account)
    except apitools_exceptions.HttpConflictError:
        raise AppAlreadyExistsError('The project [{project}] already contains an App Engine application. You can deploy your application using `gcloud app deploy`.'.format(project=project))