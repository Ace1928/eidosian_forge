from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.web_security_scanner import wss_base
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions
def _RequireAllFlagsOrRaiseForAuthType(args, flags, auth_type):
    argvars = vars(args)
    for flag in flags:
        if argvars[flag] is None:
            dashed = flag.replace('_', '-')
            raise exceptions.RequiredArgumentException('--{0}'.format(dashed), 'Required by --auth-type={0}'.format(auth_type))