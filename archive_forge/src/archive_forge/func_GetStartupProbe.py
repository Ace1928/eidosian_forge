from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import json
import textwrap
from typing import Mapping
from googlecloudsdk.api_lib.run import container_resource
from googlecloudsdk.api_lib.run import k8s_object
from googlecloudsdk.core.console import console_attr
from googlecloudsdk.core.resource import custom_printer_base as cp
def GetStartupProbe(container: container_resource.Container, labels: Mapping[str, str], is_primary: bool) -> str:
    probe_type = ''
    if is_primary:
        probe_type = labels.get('run.googleapis.com/startupProbeType', '')
    return _GetProbe(container.startupProbe, probe_type)