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
def _create_idp_sso_descriptor(self):

    def get_cert():
        try:
            return sigver.read_cert_from_file(CONF.saml.certfile, 'pem')
        except (IOError, sigver.CertificateError) as e:
            msg = 'Cannot open certificate %(cert_file)s.Reason: %(reason)s' % {'cert_file': CONF.saml.certfile, 'reason': e}
            tr_msg = _('Cannot open certificate %(cert_file)s.Reason: %(reason)s') % {'cert_file': CONF.saml.certfile, 'reason': e}
            LOG.error(msg)
            raise IOError(tr_msg)

    def key_descriptor():
        cert = get_cert()
        return md.KeyDescriptor(key_info=xmldsig.KeyInfo(x509_data=xmldsig.X509Data(x509_certificate=xmldsig.X509Certificate(text=cert))), use='signing')

    def single_sign_on_service():
        idp_sso_endpoint = CONF.saml.idp_sso_endpoint
        return md.SingleSignOnService(binding=saml2.BINDING_URI, location=idp_sso_endpoint)

    def organization():
        name = md.OrganizationName(lang=CONF.saml.idp_lang, text=CONF.saml.idp_organization_name)
        display_name = md.OrganizationDisplayName(lang=CONF.saml.idp_lang, text=CONF.saml.idp_organization_display_name)
        url = md.OrganizationURL(lang=CONF.saml.idp_lang, text=CONF.saml.idp_organization_url)
        return md.Organization(organization_display_name=display_name, organization_url=url, organization_name=name)

    def contact_person():
        company = md.Company(text=CONF.saml.idp_contact_company)
        given_name = md.GivenName(text=CONF.saml.idp_contact_name)
        surname = md.SurName(text=CONF.saml.idp_contact_surname)
        email = md.EmailAddress(text=CONF.saml.idp_contact_email)
        telephone = md.TelephoneNumber(text=CONF.saml.idp_contact_telephone)
        contact_type = CONF.saml.idp_contact_type
        return md.ContactPerson(company=company, given_name=given_name, sur_name=surname, email_address=email, telephone_number=telephone, contact_type=contact_type)

    def name_id_format():
        return md.NameIDFormat(text=saml.NAMEID_FORMAT_TRANSIENT)
    idpsso = md.IDPSSODescriptor()
    idpsso.protocol_support_enumeration = samlp.NAMESPACE
    idpsso.key_descriptor = key_descriptor()
    idpsso.single_sign_on_service = single_sign_on_service()
    idpsso.name_id_format = name_id_format()
    if self._check_organization_values():
        idpsso.organization = organization()
    if self._check_contact_person_values():
        idpsso.contact_person = contact_person()
    return idpsso