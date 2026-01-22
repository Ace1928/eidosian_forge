import json
import logging
import time
from oauth2client import _helpers
from oauth2client import _pure_python_crypt
def _check_audience(payload_dict, audience):
    """Checks audience field from a JWT payload.

    Does nothing if the passed in ``audience`` is null.

    Args:
        payload_dict: dict, A dictionary containing a JWT payload.
        audience: string or NoneType, an audience to check for in
                  the JWT payload.

    Raises:
        AppIdentityError: If there is no ``'aud'`` field in the payload
                          dictionary but there is an ``audience`` to check.
        AppIdentityError: If the ``'aud'`` field in the payload dictionary
                          does not match the ``audience``.
    """
    if audience is None:
        return
    audience_in_payload = payload_dict.get('aud')
    if audience_in_payload is None:
        raise AppIdentityError('No aud field in token: {0}'.format(payload_dict))
    if audience_in_payload != audience:
        raise AppIdentityError('Wrong recipient, {0} != {1}: {2}'.format(audience_in_payload, audience, payload_dict))