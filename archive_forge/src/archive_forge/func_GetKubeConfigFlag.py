from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import os
from googlecloudsdk.api_lib.container import kubeconfig as kconfig
from googlecloudsdk.calliope import base
from googlecloudsdk.core import exceptions as core_exceptions
def GetKubeConfigFlag(help_txt='The path to the Kubeconfig file to use.', required=False):
    return base.Argument('--kubeconfig', required=required, help=help_txt)