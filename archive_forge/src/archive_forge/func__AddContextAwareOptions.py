from __future__ import absolute_import
from __future__ import unicode_literals
import json
import os
import bootstrapping
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.core import config
from googlecloudsdk.core import context_aware
from googlecloudsdk.core import log
from googlecloudsdk.core import metrics
from googlecloudsdk.core import properties
from googlecloudsdk.core.credentials import gce as c_gce
from googlecloudsdk.core.util import encoding
from googlecloudsdk.core.util import files
def _AddContextAwareOptions(args):
    """Adds device certificate settings for mTLS."""
    context_config = context_aware.Config()
    if context_config and context_config.config_type == context_aware.ConfigType.ENTERPRISE_CERTIFICATE:
        return
    use_client_certificate = context_config and getattr(context_config, 'use_client_certificate', True)
    _MaybeAddBotoOption(args, 'Credentials', 'use_client_certificate', use_client_certificate)
    if context_config:
        cert_provider_command = _GetCertProviderCommand(context_config)
        if isinstance(cert_provider_command, list):
            cert_provider_command = ' '.join(cert_provider_command)
        _MaybeAddBotoOption(args, 'Credentials', 'cert_provider_command', cert_provider_command)