import google.auth
import google.auth.transport.requests
from google.oauth2 import service_account

    TODO(Developer): Replace the below variables before running the code.

    *NOTE*:
    Using service account keys introduces risk; they are long-lived, and can be used by anyone
    that obtains the key. Proper rotation and storage reduce this risk but do not eliminate it.
    For these reasons, you should consider an alternative approach that
    does not use a service account key. Several alternatives to service account keys
    are described here:
    https://cloud.google.com/docs/authentication/external/set-up-adc

    Args:
        json_credential_path: Path to the service account json credential file.
        target_audience: The url or target audience to obtain the ID token for.
                        Examples: http://www.abc.com
    