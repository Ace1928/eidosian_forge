from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import exceptions as apitools_exceptions
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core.console import console_io
def CheckAppNotExists(api_client, project):
    """Raises an error if the app already exists.

  Args:
    api_client: The App Engine Admin API client
    project: The GCP project

  Raises:
    AppAlreadyExistsError if app already exists
  """
    try:
        app = api_client.GetApplication()
    except apitools_exceptions.HttpNotFoundError:
        pass
    else:
        region = ' in region [{}]'.format(app.locationId) if app.locationId else ''
        raise AppAlreadyExistsError('The project [{project}] already contains an App Engine application{region}.  You can deploy your application using `gcloud app deploy`.'.format(project=project, region=region))