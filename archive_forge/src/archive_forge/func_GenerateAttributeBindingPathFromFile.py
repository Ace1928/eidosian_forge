from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import json
import os
from apitools.base.py import encoding
from googlecloudsdk.api_lib.dataplex import util as dataplex_api
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib.iam import iam_util
from googlecloudsdk.core import yaml
from googlecloudsdk.core.util import files
import six
def GenerateAttributeBindingPathFromFile(path_file_name, path_file_format):
    """Create Path from specified file."""
    if not os.path.exists(path_file_name):
        raise exceptions.BadFileException('No such file [{0}]'.format(path_file_name))
    if os.path.isdir(path_file_name):
        raise exceptions.BadFileException('[{0}] is a directory'.format(path_file_name))
    try:
        with files.FileReader(path_file_name) as import_file:
            return ConvertPathFileToProto(import_file, path_file_format)
    except Exception as exp:
        exp_msg = getattr(exp, 'message', six.text_type(exp))
        msg = 'Unable to read Path config from specified file [{0}] because [{1}]'.format(path_file_name, exp_msg)
        raise exceptions.BadFileException(msg)