from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import logging
from googlecloudsdk.api_lib.logging import util
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core.util import platforms
def _GrpcSetupHelpMessage():
    """Returns platform-specific guidance on setup for the tail command."""
    current_os = platforms.OperatingSystem.Current()
    if current_os == platforms.OperatingSystem.WINDOWS:
        return 'The installation of the Cloud SDK is corrupted, and gRPC is inaccessible.'
    if current_os in (platforms.OperatingSystem.LINUX, platforms.OperatingSystem.MACOSX):
        return 'Please ensure that the gRPC module is installed and the environment is correctly configured. Run:\n  sudo pip3 install grpcio\nand set:\n  export CLOUDSDK_PYTHON_SITEPACKAGES=1\nFor more information, see {}'.format(_TAILING_INSTALL_LINK)
    return 'Please ensure that the gRPC module is installed and the environment is configured to allow gcloud to use the installation. For help, see {}'.format(_TAILING_INSTALL_LINK)