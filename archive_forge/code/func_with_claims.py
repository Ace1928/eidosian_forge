import copy
import datetime
import json
import cachetools
import six
from six.moves import urllib
from google.auth import _helpers
from google.auth import _service_account_info
from google.auth import crypt
from google.auth import exceptions
import google.auth.credentials
def with_claims(self, issuer=None, subject=None, additional_claims=None):
    """Returns a copy of these credentials with modified claims.

        Args:
            issuer (str): The `iss` claim. If unspecified the current issuer
                claim will be used.
            subject (str): The `sub` claim. If unspecified the current subject
                claim will be used.
            additional_claims (Mapping[str, str]): Any additional claims for
                the JWT payload. This will be merged with the current
                additional claims.

        Returns:
            google.auth.jwt.OnDemandCredentials: A new credentials instance.
        """
    new_additional_claims = copy.deepcopy(self._additional_claims)
    new_additional_claims.update(additional_claims or {})
    return self.__class__(self._signer, issuer=issuer if issuer is not None else self._issuer, subject=subject if subject is not None else self._subject, additional_claims=new_additional_claims, max_cache_size=self._cache.maxsize, quota_project_id=self._quota_project_id)