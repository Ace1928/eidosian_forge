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
def _create_signature(self):
    """Create an object that represents a SAML <Signature>.

        This must be filled with algorithms that the signing binary will apply
        in order to sign the whole message.
        Currently we enforce X509 signing.
        Example of the template::

        <Signature xmlns="http://www.w3.org/2000/09/xmldsig#">
          <SignedInfo>
            <CanonicalizationMethod
              Algorithm="http://www.w3.org/2001/10/xml-exc-c14n#"/>
            <SignatureMethod
              Algorithm="http://www.w3.org/2000/09/xmldsig#rsa-sha1"/>
            <Reference URI="#<Assertion ID>">
              <Transforms>
                <Transform
            Algorithm="http://www.w3.org/2000/09/xmldsig#enveloped-signature"/>
               <Transform Algorithm="http://www.w3.org/2001/10/xml-exc-c14n#"/>
              </Transforms>
             <DigestMethod Algorithm="http://www.w3.org/2000/09/xmldsig#sha1"/>
             <DigestValue />
            </Reference>
          </SignedInfo>
          <SignatureValue />
          <KeyInfo>
            <X509Data />
          </KeyInfo>
        </Signature>

        :returns: XML <Signature> object

        """
    canonicalization_method = xmldsig.CanonicalizationMethod()
    if hasattr(xmldsig, 'TRANSFORM_C14N'):
        canonicalization_method.algorithm = xmldsig.TRANSFORM_C14N
    else:
        canonicalization_method.algorithm = xmldsig.ALG_EXC_C14N
    signature_method = xmldsig.SignatureMethod(algorithm=xmldsig.SIG_RSA_SHA1)
    transforms = xmldsig.Transforms()
    envelope_transform = xmldsig.Transform(algorithm=xmldsig.TRANSFORM_ENVELOPED)
    if hasattr(xmldsig, 'TRANSFORM_C14N'):
        c14_transform = xmldsig.Transform(algorithm=xmldsig.TRANSFORM_C14N)
    else:
        c14_transform = xmldsig.Transform(algorithm=xmldsig.ALG_EXC_C14N)
    transforms.transform = [envelope_transform, c14_transform]
    digest_method = xmldsig.DigestMethod(algorithm=xmldsig.DIGEST_SHA1)
    digest_value = xmldsig.DigestValue()
    reference = xmldsig.Reference()
    reference.uri = '#' + self.assertion_id
    reference.digest_method = digest_method
    reference.digest_value = digest_value
    reference.transforms = transforms
    signed_info = xmldsig.SignedInfo()
    signed_info.canonicalization_method = canonicalization_method
    signed_info.signature_method = signature_method
    signed_info.reference = reference
    key_info = xmldsig.KeyInfo()
    key_info.x509_data = xmldsig.X509Data()
    signature = xmldsig.Signature()
    signature.signed_info = signed_info
    signature.signature_value = xmldsig.SignatureValue()
    signature.key_info = key_info
    return signature