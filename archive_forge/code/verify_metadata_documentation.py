import argparse
from saml2.attribute_converter import ac_factory
from saml2.httpbase import HTTPBase
from saml2.mdstore import MetaDataExtern
from saml2.mdstore import MetaDataFile
from saml2.sigver import SecurityContext
from saml2.sigver import _get_xmlsec_cryptobackend

A script that imports and verifies metadata.
