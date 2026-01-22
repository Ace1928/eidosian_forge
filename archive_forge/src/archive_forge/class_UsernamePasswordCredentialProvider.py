from typing import Optional, Tuple, Union
class UsernamePasswordCredentialProvider(CredentialProvider):
    """
    Simple implementation of CredentialProvider that just wraps static
    username and password.
    """

    def __init__(self, username: Optional[str]=None, password: Optional[str]=None):
        self.username = username or ''
        self.password = password or ''

    def get_credentials(self):
        if self.username:
            return (self.username, self.password)
        return (self.password,)