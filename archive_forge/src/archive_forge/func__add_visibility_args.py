import argparse
from base64 import b64encode
import logging
import os
import sys
from cinderclient import api_versions
from openstack import exceptions as sdk_exceptions
from openstack.image import image_signer
from osc_lib.api import utils as api_utils
from osc_lib.cli import format_columns
from osc_lib.cli import parseractions
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.common import pagination
from openstackclient.common import progressbar
from openstackclient.i18n import _
from openstackclient.identity import common as identity_common
def _add_visibility_args(parser):
    public_group = parser.add_mutually_exclusive_group()
    public_group.add_argument('--public', action='store_const', const='public', dest='visibility', help=_('Image is accessible and visible to all users'))
    public_group.add_argument('--private', action='store_const', const='private', dest='visibility', help=_('Image is only accessible by the owner (default until --os-image-api-version 2.5)'))
    public_group.add_argument('--community', action='store_const', const='community', dest='visibility', help=_('Image is accessible by all users but does not appear in the default image list of any user except the owner (requires --os-image-api-version 2.5 or later)'))
    public_group.add_argument('--shared', action='store_const', const='shared', dest='visibility', help=_('Image is only accessible by the owner and image members (requires --os-image-api-version 2.5 or later) (default since --os-image-api-version 2.5)'))