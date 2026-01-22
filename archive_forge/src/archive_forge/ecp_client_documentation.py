from http import cookiejar as cookielib
import logging
from saml2 import BINDING_SOAP
from saml2 import SAMLError
from saml2 import saml
from saml2 import samlp
from saml2 import soap
from saml2.client_base import MIME_PAOS
from saml2.config import Config
from saml2.entity import Entity
from saml2.httpbase import dict2set_list
from saml2.httpbase import set_list2dict
from saml2.mdstore import MetadataStore
from saml2.profile import ecp
from saml2.profile import paos
from saml2.s_utils import BadRequest

        This is the method that should be used by someone that wants
        to authenticate using SAML ECP

        :param url: The page that access is sought for
        :param idp_entity_id: The entity ID of the IdP that should be
            used for authentication
        :param op: Which HTTP operation (GET/POST/PUT/DELETE)
        :param opargs: Arguments to the HTTP call
        :return: The page
        