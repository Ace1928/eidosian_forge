from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.asset import client_util
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
def AddAssetTypesArgument(parser):
    parser.add_argument('--asset-types', metavar='ASSET_TYPES', type=arg_parsers.ArgList(), default=[], help="        List of asset types that the IAM policies are attached to. If empty, it\n        will search the IAM policies that are attached to all the [searchable asset types](https://cloud.google.com/asset-inventory/docs/supported-asset-types).\n\n        Regular expressions are also supported. For example:\n\n          * ``compute.googleapis.com.*'' snapshots IAM policies attached to\n            asset type starts with ``compute.googleapis.com''.\n          * ``.*Instance'' snapshots IAM policies attached to asset type ends\n            with ``Instance''.\n          * ``.*Instance.*'' snapshots IAM policies attached to asset type\n            contains ``Instance''.\n\n        See [RE2](https://github.com/google/re2/wiki/Syntax) for all supported\n        regular expression syntax. If the regular expression does not match any\n        supported asset type, an ``INVALID_ARGUMENT'' error will be returned.\n        ")