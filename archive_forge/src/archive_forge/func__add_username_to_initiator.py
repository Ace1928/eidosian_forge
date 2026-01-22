import collections
import functools
import inspect
import socket
import flask
from oslo_log import log
import oslo_messaging
from oslo_utils import reflection
import pycadf
from pycadf import cadftaxonomy as taxonomy
from pycadf import cadftype
from pycadf import credential
from pycadf import eventfactory
from pycadf import host
from pycadf import reason
from pycadf import resource
from keystone.common import context
from keystone.common import provider_api
from keystone.common import utils
import keystone.conf
from keystone import exception
from keystone.i18n import _
def _add_username_to_initiator(initiator):
    """Add the username to the initiator if missing."""
    if hasattr(initiator, 'username'):
        return initiator
    try:
        user_ref = PROVIDERS.identity_api.get_user(initiator.user_id)
        initiator.username = user_ref['name']
    except (exception.UserNotFound, AttributeError):
        pass
    return initiator