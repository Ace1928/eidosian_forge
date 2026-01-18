import google
import google.auth.transport.requests
from google.oauth2 import id_token
def verify_google_idtoken(idtoken: str, audience='iap.googleapis.com', jwk_url='https://www.googleapis.com/oauth2/v3/certs'):
    """
      Verifies the obtained Google id token. This is done at the receiving end of the OIDC endpoint.
      The most common use case for verifying the ID token is when you are protecting
      your own APIs with IAP. Google services already verify credentials as a platform,
      so verifying ID tokens before making Google API calls is usually unnecessary.

    Args:
        idtoken: The Google ID token to verify.

        audience: The service name for which the id token is requested. Service name refers to the
            logical identifier of an API service, such as "iap.googleapis.com".

        jwk_url: To verify id tokens, get the Json Web Key endpoint (jwk).
            OpenID Connect allows the use of a "Discovery document," a JSON document found at a
            well-known location containing key-value pairs which provide details about the
            OpenID Connect provider's configuration.
            For more information on validating the jwt, see:
            https://developers.google.com/identity/protocols/oauth2/openid-connect#validatinganidtoken

            Here, we validate Google's token using Google's OpenID Connect service (jwkUrl).
            For more information on jwk,see:
            https://auth0.com/docs/secure/tokens/json-web-tokens/json-web-key-sets
    """
    request = google.auth.transport.requests.Request()
    result = id_token.verify_token(idtoken, request, audience, clock_skew_in_seconds=10)
    if not result['sub'] is None:
        print(f'User id: {result['sub']}')
    if result['email_verified'] == 'True':
        print(f'Email verified {result['email']}')