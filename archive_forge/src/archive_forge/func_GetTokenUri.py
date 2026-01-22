from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import abc
import json
import textwrap
from googlecloudsdk.command_lib.util import check_browser
from googlecloudsdk.core import config
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core import yaml
from googlecloudsdk.core.console import console_io
from googlecloudsdk.core.util import files
import six
def GetTokenUri():
    """Get context dependent Token URI."""
    if properties.VALUES.context_aware.use_client_certificate.GetBool():
        token_uri = properties.VALUES.auth.mtls_token_host.Get(required=True)
    else:
        token_uri = properties.VALUES.auth.token_host.Get(required=True)
    return token_uri