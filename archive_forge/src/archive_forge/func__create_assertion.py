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
def _create_assertion(self, issuer, signature, subject, authn_statement, attribute_statement):
    """Create an object that represents a SAML Assertion.

        <ns0:Assertion
          ID="35daed258ba647ba8962e9baff4d6a46"
          IssueInstant="2014-06-11T15:45:58Z"
          Version="2.0">
            <ns0:Issuer> ... </ns0:Issuer>
            <ns1:Signature> ... </ns1:Signature>
            <ns0:Subject> ... </ns0:Subject>
            <ns0:AuthnStatement> ... </ns0:AuthnStatement>
            <ns0:AttributeStatement> ... </ns0:AttributeStatement>
        </ns0:Assertion>

        :returns: XML <Assertion> object

        """
    assertion = saml.Assertion()
    assertion.id = self.assertion_id
    assertion.issue_instant = utils.isotime()
    assertion.version = '2.0'
    assertion.issuer = issuer
    assertion.signature = signature
    assertion.subject = subject
    assertion.authn_statement = authn_statement
    assertion.attribute_statement = attribute_statement
    return assertion