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
def ModifyUpdateAuthorizedViewRequest(original_ref, args, req):
    """Parse argument and construct update authorized view request.

  Args:
    original_ref: the gcloud resource.
    args: input arguments.
    req: the real request to be sent to backend service.

  Returns:
    The real request to be sent to backend service.
  """
    current_authorized_view = None
    if args.definition_file:
        req.authorizedView = ParseAuthorizedViewFromYamlOrJsonDefinitionFile(args.definition_file, args.pre_encoded)
    else:
        current_authorized_view = GetCurrentAuthorizedView(original_ref.RelativeName(), not args.pre_encoded)
        req.authorizedView = PromptForAuthorizedViewDefinition(is_create=False, pre_encoded=args.pre_encoded, current_authorized_view=current_authorized_view)
    if req.authorizedView.subsetView is not None:
        req = AddFieldToUpdateMask('subset_view', req)
    if req.authorizedView.deletionProtection is not None:
        req = AddFieldToUpdateMask('deletion_protection', req)
    if args.interactive:
        if current_authorized_view is None:
            current_authorized_view = GetCurrentAuthorizedView(original_ref.RelativeName(), check_ascii=False)
        new_authorized_view = copy.deepcopy(current_authorized_view)
        if req.authorizedView.subsetView is not None:
            new_authorized_view.subsetView = req.authorizedView.subsetView
        if req.authorizedView.deletionProtection is not None:
            new_authorized_view.deletionProtection = req.authorizedView.deletionProtection
        buf = io.StringIO()
        differ = resource_diff.ResourceDiff(original=current_authorized_view, changed=new_authorized_view)
        differ.Print('default', out=buf)
        if buf.getvalue():
            console_io.PromptContinue(message='Difference between the current authorized view and the new authorized view:\n' + buf.getvalue(), cancel_on_no=True)
        else:
            console_io.PromptContinue(message='The authorized view will NOT change with this update.', cancel_on_no=True)
    req.authorizedView.name = None
    if args.ignore_warnings:
        req.ignoreWarnings = True
    return req