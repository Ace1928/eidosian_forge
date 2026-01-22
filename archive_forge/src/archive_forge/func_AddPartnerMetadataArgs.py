from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import json
from apitools.base.protorpclite import protojson
from apitools.base.py import encoding
from apitools.base.py import extra_types
from googlecloudsdk.api_lib.compute import exceptions
from googlecloudsdk.api_lib.util import apis as core_apis
from googlecloudsdk.calliope import arg_parsers
def AddPartnerMetadataArgs(parser):
    """Adds --metadata and --metadata-from-file flags."""
    parser.add_argument('--partner-metadata', type=arg_parsers.ArgDict(), help='Partner Metadata assigned to the instance. A map from a subdomain(namespace) to entries map.', default={}, metavar='KEY=VALUE', action=arg_parsers.UpdateAction)
    parser.add_argument('--partner-metadata-from-file', type=arg_parsers.FileContents(), help='path to json local file which including the definintion of partner metadata.', metavar='LOCAL_FILE_PATH')