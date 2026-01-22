from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.web_security_scanner import wss_base
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions
def _PasswordFlag():
    """Returns a flag for setting an auth password."""
    return base.Argument('--auth-password', help="      Password for the test user. Required if `--auth-type` is not 'none'.\n      ")