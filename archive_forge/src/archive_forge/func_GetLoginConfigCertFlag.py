from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.core.util import files
def GetLoginConfigCertFlag():
    return base.Argument('--login-config-cert', required=False, type=ExpandLocalDirAndVersion, help='Specifies the CA certificate file to be added to trusted pool for making HTTPS connections to a `--login-config` URL.')