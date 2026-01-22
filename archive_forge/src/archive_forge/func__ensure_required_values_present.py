import datetime
import os
import subprocess  # nosec : see comments in the code below
import uuid
from oslo_log import log
from oslo_utils import fileutils
from oslo_utils import importutils
from oslo_utils import timeutils
import saml2
from saml2 import client_base
from saml2 import md
from saml2.profile import ecp
from saml2 import saml
from saml2 import samlp
from saml2.schema import soapenv
from saml2 import sigver
from keystone.common import utils
import keystone.conf
from keystone import exception
from keystone.i18n import _
def _ensure_required_values_present(self):
    """Ensure idp_sso_endpoint and idp_entity_id have values."""
    if CONF.saml.idp_entity_id is None:
        msg = _('Ensure configuration option idp_entity_id is set.')
        raise exception.ValidationError(msg)
    if CONF.saml.idp_sso_endpoint is None:
        msg = _('Ensure configuration option idp_sso_endpoint is set.')
        raise exception.ValidationError(msg)