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
def GenerateDataAccessSpec(args):
    """Generate Data Access Spec From Arguments."""
    module = dataplex_api.GetMessageModule()
    data_access_spec = module.GoogleCloudDataplexV1DataAccessSpec(readers=args.data_readers)
    return data_access_spec