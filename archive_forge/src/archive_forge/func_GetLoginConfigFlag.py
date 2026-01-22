from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.core.util import files
def GetLoginConfigFlag():
    return base.Argument('--login-config', required=False, help='Specifies the configuration yaml file for login. Can be a file path or a URL.')