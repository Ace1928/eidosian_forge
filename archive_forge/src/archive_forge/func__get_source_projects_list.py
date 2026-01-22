import csv
import os
from apitools.base.py import exceptions as apitools_exceptions
from googlecloudsdk.api_lib.storage import insights_api
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.storage import errors
from googlecloudsdk.command_lib.storage import flags
from googlecloudsdk.command_lib.storage.insights.dataset_configs import log_util
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core.util import files
def _get_source_projects_list(source_projects_file):
    source_projects_abs_path = os.path.expanduser(source_projects_file)
    with files.FileReader(source_projects_abs_path) as f:
        try:
            reader = csv.reader(f)
            source_projects_list = []
            for row_number, row in enumerate(reader):
                row = [element.strip() for element in row if element.strip()]
                if len(row) > 1:
                    raise ValueError('Row {} Should have excatly 1 column, but found {} columns'.format(row_number, len(row)))
                if any(row) and row[0].strip():
                    try:
                        source_projects_list.append(int(row[0].strip()))
                    except ValueError:
                        raise ValueError('Source project number {} is not a valid number'.format(row[0].strip()))
        except Exception as e:
            raise errors.Error('Invalid format for file {} provided for the --source-projects-file flag.\nError: {}'.format(source_projects_file, e))
    return source_projects_list