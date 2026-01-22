from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
import re
from typing import Any
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib.compute import flags as compute_flags
from googlecloudsdk.command_lib.util.apis import arg_utils
def AddGracefulValidationArg(parser):
    help_text = "Specifies whether the request should proceed even if the\n    request includes instances that are not members of the group or that are\n    already being deleted or abandoned. By default, if you omit this flag and\n    such an instance is specified in the request, the operation fails. The\n    operation always fails if the request contains a badly formatted instance\n    name or a reference to an instance that exists in a zone or region other\n    than the group's zone or region."
    parser.add_argument('--skip-instances-on-validation-error', action='store_true', help=help_text)