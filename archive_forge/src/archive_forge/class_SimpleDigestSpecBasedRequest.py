import random
import email.message
import pyzor
class SimpleDigestSpecBasedRequest(SimpleDigestBasedRequest):

    def __init__(self, digest=None, spec=None):
        SimpleDigestBasedRequest.__init__(self, digest)
        if spec:
            flat_spec = [item for sublist in spec for item in sublist]
            self['Op-Spec'] = ','.join((str(part) for part in flat_spec))