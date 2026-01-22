from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import re
from googlecloudsdk.api_lib.container.fleet import util
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base as calliope_base
from googlecloudsdk.calliope import exceptions as calliope_exceptions
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.calliope.concepts import deps
from googlecloudsdk.command_lib.container.fleet import api_util
from googlecloudsdk.command_lib.container.fleet import util as cmd_util
from googlecloudsdk.command_lib.util.concepts import concept_parsers
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import properties
from googlecloudsdk.core.console import console_io
def InProdRegionalAllowlist(project, track=None):
    """Returns whether project is allowlisted for regional memberships in Prod.

  This will be updated as regionalization is released, and eventually deleted
  when it is fully rolled out.

  Args:
     project: The parent project ID of the membership
    track: The release track of the command

  Returns:
    A bool, whether project is allowlisted for regional memberships in Prod
  """
    prod_regional_allowlist = ['gkeconnect-prober', 'gkeconnect-e2e', 'gkehub-cep-test', 'connectgateway-gke-testing', 'xuebinz-gke', 'kolber-anthos-testing', 'anthonytong-hub2', 'wenjuntoy2', 'hub-regionalisation-test', 'hub-regionalisation-test-2', 'a4vm-ui-tests-3', 'm4a-ui-playground-1', 'anthos-cl-e2e-tests', 'a4vm-ui-playground', 'm4a-ui-playground-1']
    return track is calliope_base.ReleaseTrack.ALPHA and project in prod_regional_allowlist