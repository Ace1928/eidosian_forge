import io
import json
import os
from google.auth import _helpers
from google.auth import exceptions
from google.auth import external_account
Creates an IdentityPool Credentials instance from an external account json file.

        Args:
            filename (str): The path to the IdentityPool external account json file.
            kwargs: Additional arguments to pass to the constructor.

        Returns:
            google.auth.identity_pool.Credentials: The constructed
                credentials.
        