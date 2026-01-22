from subprocess import PIPE
from subprocess import Popen
from saml2.extension.algsupport import DigestMethod
from saml2.extension.algsupport import SigningMethod
from saml2.sigver import get_xmlsec_binary
def algorithm_support_in_metadata(xmlsec):
    if xmlsec is None:
        return []
    support = get_algorithm_support(xmlsec)
    element_list = []
    for alg in support['digest']:
        element_list.append(DigestMethod(algorithm=DIGEST_METHODS[alg]))
    for alg in support['signing']:
        element_list.append(SigningMethod(algorithm=SIGNING_METHODS[alg]))
    return element_list