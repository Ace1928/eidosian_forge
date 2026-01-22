from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import ipaddress
import re
from typing import Any
from googlecloudsdk.calliope import exceptions
def _IsNetworkConfigPresentInService(service):
    return service.networkConfig and service.networkConfig.consumers