from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import base64
import binascii
import copy
import io
import json
import textwrap
from apitools.base.py import encoding
from apitools.base.py import exceptions as api_exceptions
from googlecloudsdk.api_lib.bigtable import util
from googlecloudsdk.api_lib.util import exceptions
from googlecloudsdk.calliope import exceptions as calliope_exceptions
from googlecloudsdk.core import yaml
from googlecloudsdk.core.console import console_io
from googlecloudsdk.core.resource import resource_diff
from googlecloudsdk.core.util import edit
import six
def ModifyCreateAuthorizedViewRequest(unused_ref, args, req):
    """Parse argument and construct create authorized view request.

  Args:
    unused_ref: the gcloud resource (unused).
    args: input arguments.
    req: the real request to be sent to backend service.

  Returns:
    The real request to be sent to backend service.
  """
    if args.definition_file:
        req.authorizedView = ParseAuthorizedViewFromYamlOrJsonDefinitionFile(args.definition_file, args.pre_encoded)
    else:
        req.authorizedView = PromptForAuthorizedViewDefinition(is_create=True, pre_encoded=args.pre_encoded)
    req.authorizedView.name = None
    return req