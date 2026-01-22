from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import json
import os
import enum
from googlecloudsdk.command_lib.container.binauthz import exceptions
from googlecloudsdk.core import yaml
from googlecloudsdk.core.util import files
import six
def LoadResourceFile(input_fname):
    """Load an input resource file in either JSON or YAML format.

  Args:
    input_fname: The name of the file to convert to parse.

  Returns:
    The Python object resulting from the decode.

  Raises:
    ResourceFileReadError: An error occurred attempting to read the input file.
    ResourceFileTypeError: The input file was an unsupported type.
    ResourceFileParseError: A parse error occurred.
  """
    try:
        input_text = files.ReadFileContents(input_fname)
    except files.Error as e:
        raise ResourceFileReadError(six.text_type(e))
    file_type = GetResourceFileType(input_fname)
    if file_type == ResourceFileType.JSON:
        try:
            return json.loads(input_text)
        except ValueError as e:
            raise ResourceFileParseError('Error in resource file JSON: ' + six.text_type(e))
    elif file_type == ResourceFileType.YAML:
        try:
            return yaml.load(input_text)
        except yaml.YAMLParseError as e:
            raise ResourceFileParseError('Error in resource file YAML: ' + six.text_type(e))
    else:
        raise ResourceFileTypeError('Input file [{}] not of type YAML or JSON'.format(input_fname))