from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import argparse
import io
import os
import re
from googlecloudsdk.api_lib.compute import base_classes
from googlecloudsdk.api_lib.compute import lister
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib.compute import ssh_utils
from googlecloudsdk.command_lib.compute.diagnose import external_helper
from googlecloudsdk.command_lib.compute.diagnose import internal_helpers
from googlecloudsdk.command_lib.util.ssh import ssh
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core.console import console_io
from googlecloudsdk.core.util import files
import six
@classmethod
def ObtainInstances(cls, names, **kwargs):
    """Returns a list of instances according to the flags."""
    errors = []
    result = lister.GetZonalResources(service=kwargs['service'], project=kwargs['project'], requested_zones=kwargs['zones'], filter_expr=kwargs['filters'], http=kwargs['http'], batch_url=kwargs['batch_url'], errors=errors)
    instances = list(result)
    filtered_instances = []
    if not names:
        filtered_instances = instances
    else:
        for name in names:
            name_match = None
            in_name = None
            in_self_link = None
            for instance in instances:
                if name == instance.name:
                    name_match = instance
                    break
                elif name in instance.name:
                    in_name = instance
                elif name in instance.selfLink:
                    in_self_link = instance
            if name_match:
                filtered_instances.append(name_match)
            elif in_name:
                filtered_instances.append(in_name)
            elif in_self_link:
                filtered_instances.append(in_self_link)
    return filtered_instances