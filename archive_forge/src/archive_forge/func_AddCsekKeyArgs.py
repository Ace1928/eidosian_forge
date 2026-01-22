from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import abc
import base64
import json
import re
from googlecloudsdk.api_lib.compute import exceptions
from googlecloudsdk.core import resources
from googlecloudsdk.core.console import console_io
import six
def AddCsekKeyArgs(parser, flags_about_creation=True, resource_type='resource'):
    """Adds arguments related to csek keys."""
    parser.add_argument('--csek-key-file', metavar='FILE', help='      Path to a Customer-Supplied Encryption Key (CSEK) key file that maps\n      Compute Engine {resource}s to user managed keys to be used when\n      creating, mounting, or taking snapshots of disks.\n\n      If you pass `-` as value of the flag, the CSEK is read from stdin.\n      See {csek_help} for more details.\n      '.format(resource=resource_type, csek_help=CSEK_HELP_URL))
    if flags_about_creation:
        parser.add_argument('--require-csek-key-create', action='store_true', default=True, help='        Refuse to create {resource}s not protected by a user managed key in\n        the key file when --csek-key-file is given. This behavior is enabled\n        by default to prevent incorrect gcloud invocations from accidentally\n        creating {resource}s with no user managed key. Disabling the check\n        allows creation of some {resource}s without a matching\n        Customer-Supplied Encryption Key in the supplied --csek-key-file.\n        See {csek_help} for more details.\n        '.format(resource=resource_type, csek_help=CSEK_HELP_URL))