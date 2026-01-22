import json
import logging
import time
from oauth2client import _helpers
from oauth2client import _pure_python_crypt
Verify a JWT against public certs.

    See http://self-issued.info/docs/draft-jones-json-web-token.html.

    Args:
        jwt: string, A JWT.
        certs: dict, Dictionary where values of public keys in PEM format.
        audience: string, The audience, 'aud', that this JWT should contain. If
                  None then the JWT's 'aud' parameter is not verified.

    Returns:
        dict, The deserialized JSON payload in the JWT.

    Raises:
        AppIdentityError: if any checks are failed.
    