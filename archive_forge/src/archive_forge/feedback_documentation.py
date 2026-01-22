from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import datetime
import textwrap
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib import feedback_util
from googlecloudsdk.command_lib import info_holder
from googlecloudsdk.core import log
from googlecloudsdk.core.console import console_io
from googlecloudsdk.core.util import files
from googlecloudsdk.core.util import text as text_util
import six
from six.moves import map
Provide feedback to the Google Cloud CLI team.

  The Google Cloud CLI team offers support through a number of channels:

  * Google Cloud CLI Issue Tracker
  * Stack Overflow "#gcloud" tag
  * google-cloud-dev Google group

  This command lists the available channels and facilitates getting help through
  one of them by opening a web browser to the relevant page, possibly with
  information relevant to the current install and configuration pre-populated in
  form fields on that page.
  