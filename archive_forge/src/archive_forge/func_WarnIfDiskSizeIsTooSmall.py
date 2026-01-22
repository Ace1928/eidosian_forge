from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import argparse
import io
import re
from googlecloudsdk.api_lib.compute import constants
from googlecloudsdk.api_lib.compute import exceptions
from googlecloudsdk.calliope import exceptions as calliope_exceptions
from googlecloudsdk.command_lib.compute import exceptions as compute_exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core.console import console_io
from googlecloudsdk.core.resource import resource_printer
import ipaddr
import six
def WarnIfDiskSizeIsTooSmall(size_gb, disk_type):
    """Writes a warning message if the given disk size is too small."""
    if not size_gb:
        return
    if disk_type and (constants.DISK_TYPE_PD_BALANCED in disk_type or constants.DISK_TYPE_PD_SSD in disk_type or constants.DISK_TYPE_PD_EXTREME in disk_type):
        warning_threshold_gb = constants.SSD_DISK_PERFORMANCE_WARNING_GB
    elif disk_type and (constants.DISK_TYPE_HD_EXTREME in disk_type or constants.DISK_TYPE_HD_BALANCED in disk_type or constants.DISK_TYPE_HD_THROUGHPUT in disk_type):
        warning_threshold_gb = 0
    else:
        warning_threshold_gb = constants.STANDARD_DISK_PERFORMANCE_WARNING_GB
    if size_gb < warning_threshold_gb:
        log.warning(WARN_IF_DISK_SIZE_IS_TOO_SMALL, warning_threshold_gb)