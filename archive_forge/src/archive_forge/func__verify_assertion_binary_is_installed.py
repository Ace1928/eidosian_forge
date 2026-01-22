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
def _verify_assertion_binary_is_installed():
    """Make sure the specified xmlsec binary is installed.

    If the binary specified in configuration isn't installed, make sure we
    leave some sort of useful error message for operators since the absense of
    it is going to throw an HTTP 500.

    """
    try:
        subprocess.check_output(['/usr/bin/which', CONF.saml.xmlsec1_binary])
    except subprocess.CalledProcessError:
        msg = 'Unable to locate %(binary)s binary on the system. Check to make sure it is installed.' % {'binary': CONF.saml.xmlsec1_binary}
        tr_msg = _('Unable to locate %(binary)s binary on the system. Check to make sure it is installed.') % {'binary': CONF.saml.xmlsec1_binary}
        LOG.error(msg)
        raise exception.SAMLSigningError(reason=tr_msg)