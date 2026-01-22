from collections import namedtuple
from .agent import AgentKey
from .util import get_logger
from .ssh_exception import AuthenticationException
class InMemoryPrivateKey(PrivateKey):
    """
    An in-memory, decrypted `.PKey` object.
    """

    def __init__(self, username, pkey):
        super().__init__(username=username)
        self.pkey = pkey

    def __repr__(self):
        rep = super()._repr(pkey=self.pkey)
        if isinstance(self.pkey, AgentKey):
            rep += ' [agent]'
        return rep