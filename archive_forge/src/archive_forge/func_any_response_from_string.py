import saml2
from saml2 import SamlBase
from saml2 import saml
from saml2 import xmldsig as ds
def any_response_from_string(xmlstr):
    resp = None
    for func in [status_response_type__from_string, response_from_string, artifact_response_from_string, logout_response_from_string, name_id_mapping_response_from_string, manage_name_id_response_from_string]:
        resp = func(xmlstr)
        if resp:
            break
    if not resp:
        raise Exception('Unknown response type')
    return resp