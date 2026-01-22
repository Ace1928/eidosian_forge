import calendar
import time
import urllib
from cryptography.hazmat import backends
from cryptography.hazmat.primitives import serialization
from cryptography import x509 as cryptography_x509
from keystoneauth1 import identity
from keystoneauth1 import loading
from keystoneauth1 import service_token
from keystoneauth1 import session
from oslo_config import cfg
from oslo_log import log as logging
from oslo_utils import excutils
from castellan.common import exception
from castellan.common.objects import key as key_base_class
from castellan.common.objects import opaque_data as op_data
from castellan.i18n import _
from castellan.key_manager import key_manager
from barbicanclient import client as barbican_client_import
from barbicanclient import exceptions as barbican_exceptions
from oslo_utils import timeutils
Retrieves a list of managed objects that match the criteria.

        If no search criteria is given, all objects are returned.

        :param context: contains information of the user and the environment
                     for the request (castellan/context.py)
        :param object_type: the type of object to retrieve
        :param metadata_only: whether secret data should be included
        :raises KeyManagerError: if listing secrets fails
        