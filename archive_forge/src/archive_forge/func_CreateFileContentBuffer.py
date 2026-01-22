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
def CreateFileContentBuffer(messages, file_path):
    """Helper function to read file and return FileContentBuffer."""
    file_content_buffer = messages.FileContentBuffer()
    content, file_type = GetFileContentAndFileType(file_path)
    file_content_buffer.content = content
    file_content_buffer.fileType = messages.FileContentBuffer.FileTypeValueValuesEnum(file_type)
    return file_content_buffer