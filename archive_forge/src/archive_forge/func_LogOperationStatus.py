from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import base64
import binascii
import io
from googlecloudsdk.calliope import exceptions as calliope_exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core.resource import resource_printer
from googlecloudsdk.core.util import http_encoding
def LogOperationStatus(operation, operation_description):
    """Log operation warnings if there is any."""
    if operation.warnings:
        log.warning('{0} operation {1} completed with warnings:\n{2}'.format(operation_description, operation.name, RenderMessageAsYaml(operation.warnings)))
    else:
        log.status.Print('{0} operation {1} completed successfully.'.format(operation_description, operation.name))