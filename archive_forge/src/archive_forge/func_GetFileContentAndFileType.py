from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.compute import constants
from googlecloudsdk.api_lib.compute import request_helper
from googlecloudsdk.api_lib.compute import utils
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core.console import console_io
from googlecloudsdk.core.util import files as file_utils
def GetFileContentAndFileType(file_path):
    """Helper function used for read file and determine file type."""
    file_content = file_utils.ReadBinaryFileContents(file_path)
    file_type = ''
    if file_path.endswith('.bin'):
        file_type = 'BIN'
    else:
        if not IsDERForm(file_content):
            raise utils.IncorrectX509FormError('File is not in X509 binary DER form.')
        file_type = 'X509'
    return (file_content, file_type)