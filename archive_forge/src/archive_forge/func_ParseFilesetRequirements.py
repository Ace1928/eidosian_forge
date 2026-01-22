from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from os import path
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from googlecloudsdk.api_lib.data_catalog import entries_v1
from googlecloudsdk.api_lib.data_catalog import util as api_util
from googlecloudsdk.command_lib.concepts import exceptions as concept_exceptions
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import yaml
from googlecloudsdk.core.util import files
from googlecloudsdk.core.util import times
import six
def ParseFilesetRequirements(ref, args, request):
    """Fileset types need a file pattern."""
    del ref
    if args.type == 'fileset' and args.gcs_file_patterns is None:
        raise concept_exceptions.ModalGroupError('gcs-file-patterns', 'type=FILESET', '--gcs-file-patterns')
    return request