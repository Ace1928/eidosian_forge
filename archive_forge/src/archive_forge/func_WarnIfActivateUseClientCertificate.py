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
def WarnIfActivateUseClientCertificate(value):
    """Warn if setting context_aware/use_client_certificate to truthy."""
    if value.lower() in ['1', 'true', 'on', 'yes', 'y']:
        mtls_not_supported_msg = 'Some services may not support client certificate authorization in this version of gcloud. When a command sends requests to such services, the requests will be executed without using a client certificate.\n\nPlease run $ gcloud topic client-certificate for more information.'
        log.warning(mtls_not_supported_msg)