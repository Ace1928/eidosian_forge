from __future__ import absolute_import, division, print_function
class GenericRestClientConfiguration(Configuration):

    def __init__(self, credential, subscription_id, credential_scopes=None, base_url=None):
        if credential is None:
            raise ValueError("Parameter 'credentials' must not be None.")
        if subscription_id is None:
            raise ValueError("Parameter 'subscription_id' must not be None.")
        if not base_url:
            base_url = 'https://management.azure.com'
        if not credential_scopes:
            credential_scopes = 'https://management.azure.com/.default'
        super(GenericRestClientConfiguration, self).__init__()
        self.credentials = credential
        self.subscription_id = subscription_id
        self.authentication_policy = BearerTokenCredentialPolicy(credential, credential_scopes)