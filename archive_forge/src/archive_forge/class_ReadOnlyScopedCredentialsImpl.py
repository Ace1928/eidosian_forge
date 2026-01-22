import datetime
import pytest  # type: ignore
from google.auth import _credentials_async as credentials
from google.auth import _helpers
class ReadOnlyScopedCredentialsImpl(credentials.ReadOnlyScoped, CredentialsImpl):

    @property
    def requires_scopes(self):
        return super(ReadOnlyScopedCredentialsImpl, self).requires_scopes