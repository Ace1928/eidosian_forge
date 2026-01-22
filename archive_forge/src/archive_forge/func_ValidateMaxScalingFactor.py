from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import ipaddress
import re
from typing import Any
from googlecloudsdk.calliope import exceptions
def ValidateMaxScalingFactor(scaling_factor):
    """Python hook to validate the max scaling factor."""
    return ValidateScalingFactorFloat(scaling_factor, '--max-scaling-factor')