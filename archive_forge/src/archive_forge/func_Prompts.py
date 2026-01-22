from __future__ import absolute_import
from __future__ import print_function
from __future__ import unicode_literals
import bootstrapping
import argparse
import os
import sys
from googlecloudsdk.calliope import actions
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.core import config
from googlecloudsdk.core import execution_utils
from googlecloudsdk.core import platforms_install
from googlecloudsdk.core import properties
from googlecloudsdk.core.console import console_io
from googlecloudsdk.core.updater import update_manager
from googlecloudsdk.core.util import encoding
from googlecloudsdk.core.util import files
from googlecloudsdk.core.util import platforms
from googlecloudsdk import gcloud_main
def Prompts(usage_reporting):
    """Display prompts to opt out of usage reporting.

  Args:
    usage_reporting: bool, If True, enable usage reporting. If None, check
    the environmental variable. If None, check if its alternate release channel.
    If not, ask.
  """
    if usage_reporting is None:
        if encoding.GetEncodedValue(os.environ, 'CLOUDSDK_CORE_DISABLE_USAGE_REPORTING') is not None:
            usage_reporting = not encoding.GetEncodedValue(os.environ, 'CLOUDSDK_CORE_DISABLE_USAGE_REPORTING')
        elif config.InstallationConfig.Load().IsAlternateReleaseChannel():
            usage_reporting = True
            print('\n    Usage reporting is always on for alternate release channels.\n    ')
        else:
            print("\nTo help improve the quality of this product, we collect anonymized usage data\nand anonymized stacktraces when crashes are encountered; additional information\nis available at <https://cloud.google.com/sdk/usage-statistics>. This data is\nhandled in accordance with our privacy policy\n<https://cloud.google.com/terms/cloud-privacy-notice>. You may choose to opt in this\ncollection now (by choosing 'Y' at the below prompt), or at any time in the\nfuture by running the following command:\n\n    gcloud config set disable_usage_reporting false\n")
            usage_reporting = console_io.PromptContinue(prompt_string='Do you want to help improve the Google Cloud CLI', default=False)
    properties.PersistProperty(properties.VALUES.core.disable_usage_reporting, not usage_reporting, scope=properties.Scope.INSTALLATION)