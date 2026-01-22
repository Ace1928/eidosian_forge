import dbm
import importlib
import logging
import shelve
import threading
from saml2 import BINDING_HTTP_REDIRECT
from saml2 import class_name
from saml2 import element_to_extension_element
from saml2 import saml
from saml2.argtree import add_path
from saml2.argtree import is_set
from saml2.assertion import Assertion
from saml2.assertion import Policy
from saml2.assertion import filter_attribute_value_assertions
from saml2.assertion import restriction_from_attribute_spec
import saml2.cryptography.symmetric
from saml2.entity import Entity
from saml2.eptid import Eptid
from saml2.eptid import EptidShelve
from saml2.ident import IdentDB
from saml2.ident import decode
from saml2.profile import ecp
from saml2.request import AssertionIDRequest
from saml2.request import AttributeQuery
from saml2.request import AuthnQuery
from saml2.request import AuthnRequest
from saml2.request import AuthzDecisionQuery
from saml2.request import NameIDMappingRequest
from saml2.s_utils import MissingValue
from saml2.s_utils import Unknown
from saml2.s_utils import rndstr
from saml2.samlp import NameIDMappingResponse
from saml2.schema import soapenv
from saml2.sdb import SessionStorage
from saml2.sigver import CertificateError
from saml2.sigver import pre_signature_part
from saml2.sigver import signed_instance_factory
def _authn_response(self, in_response_to, consumer_url, sp_entity_id, identity=None, name_id=None, status=None, authn=None, issuer=None, policy=None, sign_assertion=None, sign_response=None, best_effort=False, encrypt_assertion=False, encrypt_cert_advice=None, encrypt_cert_assertion=None, authn_statement=None, encrypt_assertion_self_contained=False, encrypted_advice_attributes=False, pefim=False, sign_alg=None, digest_alg=None, farg=None, session_not_on_or_after=None):
    """Create a response. A layer of indirection.

        :param in_response_to: The session identifier of the request
        :param consumer_url: The URL which should receive the response
        :param sp_entity_id: The entity identifier of the SP
        :param identity: A dictionary with attributes and values that are
            expected to be the bases for the assertion in the response.
        :param name_id: The identifier of the subject
        :param status: The status of the response
        :param authn: A dictionary containing information about the
            authn context.
        :param issuer: The issuer of the response
        :param policy:
        :param sign_assertion: Whether the assertion should be signed or not
        :param sign_response: Whether the response should be signed or not
        :param best_effort: Even if not the SPs demands can be met send a
            response.
        :param encrypt_assertion: True if assertions should be encrypted.
        :param encrypt_assertion_self_contained: True if all encrypted
        assertions should have alla namespaces
        selfcontained.
        :param encrypted_advice_attributes: True if assertions in the advice
        element should be encrypted.
        :param encrypt_cert_advice: Certificate to be used for encryption of
        assertions in the advice element.
        :param encrypt_cert_assertion: Certificate to be used for encryption
        of assertions.
        :param authn_statement: Authentication statement.
        :param pefim: True if a response according to the PEFIM profile
        should be created.
        :param farg: Argument to pass on to the assertion constructor
        :return: A response instance
        """
    _issuer = self._issuer(issuer)
    if pefim:
        encrypted_advice_attributes = True
        encrypt_assertion_self_contained = True
        assertion_attributes = self.setup_assertion(None, sp_entity_id, None, None, None, policy, None, None, identity, best_effort, sign_response, farg=farg)
        assertion = self.setup_assertion(authn, sp_entity_id, in_response_to, consumer_url, name_id, policy, _issuer, authn_statement, [], True, sign_response, farg=farg, session_not_on_or_after=session_not_on_or_after)
        assertion.advice = saml.Advice()
        assertion.advice.assertion.append(assertion_attributes)
    else:
        assertion = self.setup_assertion(authn, sp_entity_id, in_response_to, consumer_url, name_id, policy, _issuer, authn_statement, identity, True, sign_response, farg=farg, session_not_on_or_after=session_not_on_or_after)
    to_sign = []
    if not encrypt_assertion:
        if sign_assertion:
            sign_alg = sign_alg or self.signing_algorithm
            digest_alg = digest_alg or self.digest_algorithm
            assertion.signature = pre_signature_part(assertion.id, self.sec.my_cert, 2, sign_alg=sign_alg, digest_alg=digest_alg)
            to_sign.append((class_name(assertion), assertion.id))
    if self.support_AssertionIDRequest() or self.support_AuthnQuery():
        self.session_db.store_assertion(assertion, to_sign)
    return self._response(in_response_to, consumer_url, status, issuer, sign_response, to_sign, sp_entity_id=sp_entity_id, encrypt_assertion=encrypt_assertion, encrypt_cert_advice=encrypt_cert_advice, encrypt_cert_assertion=encrypt_cert_assertion, encrypt_assertion_self_contained=encrypt_assertion_self_contained, encrypted_advice_attributes=encrypted_advice_attributes, sign_assertion=sign_assertion, pefim=pefim, sign_alg=sign_alg, digest_alg=digest_alg, assertion=assertion)