import collections
import io
import logging
import os
from cryptography.hazmat.primitives.asymmetric import ed25519
from cryptography.hazmat.primitives import serialization
from openstack import utils as sdk_utils
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.common import pagination
from openstackclient.i18n import _
from openstackclient.identity import common as identity_common
def _get_keypair_columns(item, hide_pub_key=False, hide_priv_key=False):
    column_map = {}
    hidden_columns = ['links', 'location']
    if hide_pub_key:
        hidden_columns.append('public_key')
    if hide_priv_key:
        hidden_columns.append('private_key')
    return utils.get_osc_show_columns_for_sdk_resource(item, column_map, hidden_columns)