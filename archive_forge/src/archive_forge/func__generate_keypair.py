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
def _generate_keypair():
    """Generate a Ed25519 keypair in OpenSSH format.

    :returns: A `Keypair` named tuple with the generated private and public
    keys.
    """
    key = ed25519.Ed25519PrivateKey.generate()
    private_key = key.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.OpenSSH, serialization.NoEncryption()).decode()
    public_key = key.public_key().public_bytes(serialization.Encoding.OpenSSH, serialization.PublicFormat.OpenSSH).decode()
    return Keypair(private_key, public_key)