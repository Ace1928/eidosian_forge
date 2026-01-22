from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.iamcredentials import util as iamcred_util
from googlecloudsdk.core.credentials import store
class CredentialProvidersManager(object):
    """Context manager for handling credential provider registration."""

    def __init__(self, credential_providers=None):
        """Initializes context manager with optional credential providers.

    Args:
      credential_providers (list[object]): List of provider classes like those
        defined in core.credentials.store.py.
    """
        self._credential_providers = credential_providers

    def __enter__(self):
        """Registers sources for credentials and project for use by commands."""
        self._credential_providers = self._credential_providers or [store.GceCredentialProvider()]
        for provider in self._credential_providers:
            provider.Register()
        store.IMPERSONATION_TOKEN_PROVIDER = iamcred_util.ImpersonationAccessTokenProvider()
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        """Cleans up credential providers."""
        del [exc_type, exc_value, exc_traceback]
        for provider in self._credential_providers:
            provider.UnRegister()
        store.IMPERSONATION_TOKEN_PROVIDER = None