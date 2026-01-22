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
class ShowProjectImage(command.ShowOne):
    _description = _('Show a particular project associated with image')

    def get_parser(self, prog_name):
        parser = super().get_parser(prog_name)
        parser.add_argument('image', metavar='<image>', help=_('Image (name or ID)'))
        parser.add_argument('member', metavar='<project>', help=_('Project to show (name or ID)'))
        identity_common.add_project_domain_option_to_parser(parser)
        return parser

    def take_action(self, parsed_args):
        image_client = self.app.client_manager.image
        image = image_client.find_image(parsed_args.image, ignore_missing=False)
        obj = image_client.get_member(image=image.id, member=parsed_args.member)
        display_columns, columns = _get_member_columns(obj)
        data = utils.get_item_properties(obj, columns, formatters={})
        return (display_columns, data)