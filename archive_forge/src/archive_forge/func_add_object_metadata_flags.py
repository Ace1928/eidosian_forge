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
def add_object_metadata_flags(parser, allow_patch=False):
    """Add flags that allow setting object metadata."""
    metadata_group = parser.add_group(category='OBJECT METADATA')
    metadata_group.add_argument('--cache-control', help='How caches should handle requests and responses.')
    metadata_group.add_argument('--content-disposition', help='How content should be displayed.')
    metadata_group.add_argument('--content-encoding', help="How content is encoded (e.g. ``gzip'').")
    metadata_group.add_argument('--content-language', help='Content\'s language (e.g. ``en\'\' signifies "English").')
    metadata_group.add_argument('--content-type', help="Type of data contained in the object (e.g. ``text/html'').")
    metadata_group.add_argument('--custom-time', type=arg_parsers.Datetime.Parse, help='Custom time for Cloud Storage objects in RFC 3339 format.')
    custom_metadata_group = metadata_group.add_mutually_exclusive_group()
    custom_metadata_group.add_argument('--custom-metadata', metavar='CUSTOM_METADATA_KEYS_AND_VALUES', type=arg_parsers.ArgDict(), help='Sets custom metadata on objects. When used with `--preserve-posix`, POSIX attributes are also stored in custom metadata.')
    custom_metadata_group.add_argument('--clear-custom-metadata', action='store_true', help='Clears all custom metadata on objects. When used with `--preserve-posix`, POSIX attributes will still be stored in custom metadata.')
    update_custom_metadata_group = custom_metadata_group.add_group(help='Flags that preserve unspecified existing metadata cannot be used with `--custom-metadata` or `--clear-custom-metadata`, but can be specified together:')
    update_custom_metadata_group.add_argument('--update-custom-metadata', metavar='CUSTOM_METADATA_KEYS_AND_VALUES', type=arg_parsers.ArgDict(), help='Adds or sets individual custom metadata key value pairs on objects. Existing custom metadata not specified with this flag is not changed. This flag can be used with `--remove-custom-metadata`. When keys overlap with those provided by `--preserve-posix`, values specified by this flag are used.')
    update_custom_metadata_group.add_argument('--remove-custom-metadata', metavar='METADATA_KEYS', type=arg_parsers.ArgList(), help='Removes individual custom metadata keys from objects. This flag can be used with `--update-custom-metadata`. When used with `--preserve-posix`, POSIX attributes specified by this flag are not preserved.')
    if allow_patch:
        metadata_group.add_argument('--clear-cache-control', action='store_true', help='Clears object cache control.')
        metadata_group.add_argument('--clear-content-disposition', action='store_true', help='Clears object content disposition.')
        metadata_group.add_argument('--clear-content-encoding', action='store_true', help='Clears content encoding.')
        metadata_group.add_argument('--clear-content-language', action='store_true', help='Clears object content language.')
        metadata_group.add_argument('--clear-content-type', action='store_true', help='Clears object content type.')
        metadata_group.add_argument('--clear-custom-time', action='store_true', help='Clears object custom time.')