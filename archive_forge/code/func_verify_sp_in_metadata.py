from urllib import parse
from saml2.entity import Entity
from saml2.response import VerificationError
def verify_sp_in_metadata(self, entity_id):
    if self.metadata:
        endp = self.metadata.discovery_response(entity_id)
        if endp:
            return True
    return False