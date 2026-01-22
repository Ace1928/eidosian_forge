from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from typing import List, Optional
from apitools.base.py import encoding
from googlecloudsdk.command_lib.run.integrations.formatters import base
from googlecloudsdk.command_lib.run.integrations.formatters import states
from googlecloudsdk.core.console import console_attr
from googlecloudsdk.core.resource import custom_printer_base as cp
from googlecloudsdk.generated_clients.apis.runapps.v1alpha1 import runapps_v1alpha1_messages as runapps
def _GuessDomainFromSSLComponentName(self, name):
    parts = name.replace('d--', '').split('-')
    end_index = -2
    if parts[len(parts) - 1] != 'cert':
        end_index = -3
    return '.'.join(parts[2:end_index])