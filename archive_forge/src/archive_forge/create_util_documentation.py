from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import exceptions as apitools_exceptions
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core.console import console_io
Interactively choose a region and create an App Engine app.

  The caller is responsible for calling this method only when the user can be
  prompted interactively.

  Example interaction:

      Please choose the region where you want your App Engine application
      located:

        [1] us-east1      (supports standard and flexible)
        [2] europe-west   (supports standard)
        [3] us-central    (supports standard and flexible)
        [4] cancel
      Please enter your numeric choice:  1

  Args:
    api_client: The App Engine Admin API client
    project: The GCP project
    regions: The list of regions to choose from; if None, all possible regions
             are listed
    extra_warning: An additional warning to print before listing regions.
    service_account: The app level service account for the App Engine app.

  Raises:
    AppAlreadyExistsError if app already exists
  