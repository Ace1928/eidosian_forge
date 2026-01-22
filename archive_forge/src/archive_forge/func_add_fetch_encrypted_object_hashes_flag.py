from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import enum
from googlecloudsdk.api_lib.storage import cloud_api
from googlecloudsdk.calliope import actions
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.command_lib.storage import errors
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
def add_fetch_encrypted_object_hashes_flag(parser, is_list=True):
    """Adds flag to commands that need object hashes."""
    if is_list:
        help_text = 'API requests to the LIST endpoint do not fetch the hashes for encrypted objects by default. If this flag is set, a GET request is sent for each encrypted object in order to fetch hashes. This can significantly increase the cost of the command.'
    else:
        help_text = 'If the initial GET request returns an object encrypted with a customer-supplied encryption key, the hash fields will be null. If the matching decryption key is present on the system, this flag retries the GET request with the key.'
    parser.add_argument('--fetch-encrypted-object-hashes', action='store_true', help=help_text)