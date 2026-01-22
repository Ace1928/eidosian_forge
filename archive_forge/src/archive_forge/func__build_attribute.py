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
def _build_attribute(attribute_name, attribute_values):
    attribute = saml.Attribute()
    attribute.name = attribute_name
    for value in attribute_values:
        attribute_value = saml.AttributeValue()
        attribute_value.set_text(value)
        attribute.attribute_value.append(attribute_value)
    return attribute