from keystoneclient import exceptions as identity_exc
from keystoneclient.v3 import domains
from keystoneclient.v3 import groups
from keystoneclient.v3 import projects
from keystoneclient.v3 import users
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.i18n import _
def add_resource_option_to_parser(parser):
    enable_group = parser.add_mutually_exclusive_group()
    enable_group.add_argument('--immutable', action='store_true', help=_('Make resource immutable. An immutable project may not be deleted or modified except to remove the immutable flag'))
    enable_group.add_argument('--no-immutable', action='store_true', help=_('Make resource mutable (default)'))