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
def ResolveParentId(data_attribute_ref, args):
    if args.IsSpecified('parent') and args.parent.find('/') == -1:
        return data_attribute_ref.RelativeName().rsplit('/', 1)[0] + '/' + args.parent
    else:
        return args.parent