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
def AssertClientSecretIsInstalledType(client_id_file):
    client_type = GetClientSecretsType(client_id_file)
    if client_type != CLIENT_SECRET_INSTALLED_TYPE:
        raise InvalidClientSecretsError("Only client IDs of type '%s' are allowed, but encountered type '%s'" % (CLIENT_SECRET_INSTALLED_TYPE, client_type))