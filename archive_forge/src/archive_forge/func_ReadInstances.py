from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import io
import json
from googlecloudsdk.api_lib.ml_engine import models
from googlecloudsdk.api_lib.ml_engine import versions_api
from googlecloudsdk.core import exceptions as core_exceptions
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
from googlecloudsdk.core.console import console_io
from googlecloudsdk.core.util import encoding
import six
def ReadInstances(input_file, data_format, limit=None):
    """Reads the instances from input file.

  Args:
    input_file: An open file-like object for the input file.
    data_format: str, data format of the input file, 'json' or 'text'.
    limit: int, the maximum number of instances allowed in the file

  Returns:
    A list of instances.

  Raises:
    InvalidInstancesFileError: If the input file is invalid (invalid format or
        contains too many/zero instances).
  """
    instances = []
    for line_num, line in enumerate(input_file):
        if isinstance(line, six.binary_type):
            line = encoding.Decode(line, encoding='utf-8-sig')
        line_content = line.rstrip('\r\n')
        if not line_content:
            raise InvalidInstancesFileError('Empty line is not allowed in the instances file.')
        if limit and line_num >= limit:
            raise InvalidInstancesFileError('The gcloud CLI can currently process no more than ' + six.text_type(limit) + ' instances per file. Please use the API directly if you need to send more.')
        if data_format == 'json':
            try:
                instances.append(json.loads(line_content))
            except ValueError:
                raise InvalidInstancesFileError('Input instances are not in JSON format. See "gcloud ai-platform predict --help" for details.')
        elif data_format == 'text':
            instances.append(line_content)
    if not instances:
        raise InvalidInstancesFileError('No valid instance was found in input file.')
    return instances