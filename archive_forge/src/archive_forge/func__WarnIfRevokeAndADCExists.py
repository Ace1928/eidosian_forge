from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import os
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions as c_exc
from googlecloudsdk.command_lib.auth import auth_util
from googlecloudsdk.core import config
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core.credentials import creds as c_creds
from googlecloudsdk.core.credentials import exceptions as creds_exceptions
from googlecloudsdk.core.credentials import store as c_store
from googlecloudsdk.core.resource import resource_printer
def _WarnIfRevokeAndADCExists(creds_revoked):
    if creds_revoked and os.path.isfile(config.ADCFilePath()) and auth_util.ADCIsUserAccount():
        log.warning('You also have Application Default Credentials (ADC) set up. If you want to revoke your Application Default Credentials as well, use the `gcloud auth application-default revoke` command.\n\nFor information about ADC credentials and gcloud CLI credentials, see https://cloud.google.com/docs/authentication/external/credential-types\n')