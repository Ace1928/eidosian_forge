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
def PrintFingerprint(fingerprint):
    """Prints the fingerprint for user reference."""
    log.status.Print('The fingerprint of the deployment is %s' % base64.urlsafe_b64encode(fingerprint))