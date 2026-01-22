from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from typing import Optional
from apitools.base.py import encoding
from apitools.base.py import list_pager
from googlecloudsdk.api_lib.container.gkeonprem import client
from googlecloudsdk.api_lib.container.gkeonprem import update_mask
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.calliope import parser_extensions
from googlecloudsdk.core import properties
from googlecloudsdk.generated_clients.apis.gkeonprem.v1 import gkeonprem_v1_messages as messages
def _update_annotations(self, args: parser_extensions.Namespace):
    """Constructs proto message AnnotationsValue for update command."""
    specified_args = args.GetSpecifiedArgsDict()
    if 'add_annotations' in specified_args:
        return self._add_annotations(args)
    if 'clear_annotations' in specified_args:
        return self._clear_annotations()
    if 'remove_annotations' in specified_args:
        return self._remove_annotations(args)
    if 'set_annotations' in specified_args:
        return self._set_annotations(args)