import datetime
from google.auth import _helpers
from google.auth import _service_account_info
from google.auth import credentials
from google.auth import exceptions
from google.auth import jwt
from google.oauth2 import _client
Creates a Credentials instance from a service account json file.

        Args:
            filename (str): The path to the service account json file.
            kwargs: Additional arguments to pass to the constructor.

        Returns:
            google.oauth2.gdch_credentials.ServiceAccountCredentials: The constructed
                credentials.
        