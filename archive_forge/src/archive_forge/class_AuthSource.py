from collections import namedtuple
from .agent import AgentKey
from .util import get_logger
from .ssh_exception import AuthenticationException
class AuthSource:
    """
    Some SSH authentication source, such as a password, private key, or agent.

    See subclasses in this module for concrete implementations.

    All implementations must accept at least a ``username`` (``str``) kwarg.
    """

    def __init__(self, username):
        self.username = username

    def _repr(self, **kwargs):
        pairs = [f'{k}={v!r}' for k, v in kwargs.items()]
        joined = ', '.join(pairs)
        return f'{self.__class__.__name__}({joined})'

    def __repr__(self):
        return self._repr()

    def authenticate(self, transport):
        """
        Perform authentication.
        """
        raise NotImplementedError