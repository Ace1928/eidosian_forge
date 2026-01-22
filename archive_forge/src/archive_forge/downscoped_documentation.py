import datetime
import six
from google.auth import _helpers
from google.auth import credentials
from google.auth import exceptions
from google.oauth2 import sts
Instantiates a downscoped credentials object using the provided source
        credentials and credential access boundary rules.
        To downscope permissions of a source credential, a Credential Access Boundary
        that specifies which resources the new credential can access, as well as an
        upper bound on the permissions that are available on each resource, has to be
        defined. A downscoped credential can then be instantiated using the source
        credential and the Credential Access Boundary.

        Args:
            source_credentials (google.auth.credentials.Credentials): The source credentials
                to be downscoped based on the provided Credential Access Boundary rules.
            credential_access_boundary (google.auth.downscoped.CredentialAccessBoundary):
                The Credential Access Boundary which contains a list of access boundary
                rules. Each rule contains information on the resource that the rule applies to,
                the upper bound of the permissions that are available on that resource and an
                optional condition to further restrict permissions.
            quota_project_id (Optional[str]): The optional quota project ID.
        Raises:
            google.auth.exceptions.RefreshError: If the source credentials
                return an error on token refresh.
            google.auth.exceptions.OAuthError: If the STS token exchange
                endpoint returned an error during downscoped token generation.
        