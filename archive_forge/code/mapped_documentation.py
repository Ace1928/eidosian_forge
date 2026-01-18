import functools
import uuid
import flask
from oslo_log import log
from pycadf import cadftaxonomy as taxonomy
from urllib import parse
from keystone.auth import plugins as auth_plugins
from keystone.auth.plugins import base
from keystone.common import provider_api
from keystone import exception
from keystone.federation import constants as federation_constants
from keystone.federation import utils
from keystone.i18n import _
from keystone import notifications
Authenticate mapped user and set an authentication context.

        :param auth_payload: the content of the authentication for a
                             given method

        In addition to ``user_id`` in ``response_data``, this plugin sets
        ``group_ids``, ``OS-FEDERATION:identity_provider`` and
        ``OS-FEDERATION:protocol``

        