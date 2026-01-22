from collections import namedtuple
from .agent import AgentKey
from .util import get_logger
from .ssh_exception import AuthenticationException
class AuthResult(list):
    """
    Represents a partial or complete SSH authentication attempt.

    This class conceptually extends `AuthStrategy` by pairing the former's
    authentication **sources** with the **results** of trying to authenticate
    with them.

    `AuthResult` is a (subclass of) `list` of `namedtuple`, which are of the
    form ``namedtuple('SourceResult', 'source', 'result')`` (where the
    ``source`` member is an `AuthSource` and the ``result`` member is either a
    return value from the relevant `.Transport` method, or an exception
    object).

    .. note::
        Transport auth method results are always themselves a ``list`` of "next
        allowable authentication methods".

        In the simple case of "you just authenticated successfully", it's an
        empty list; if your auth was rejected but you're allowed to try again,
        it will be a list of string method names like ``pubkey`` or
        ``password``.

        The ``__str__`` of this class represents the empty-list scenario as the
        word ``success``, which should make reading the result of an
        authentication session more obvious to humans.

    Instances also have a `strategy` attribute referencing the `AuthStrategy`
    which was attempted.
    """

    def __init__(self, strategy, *args, **kwargs):
        self.strategy = strategy
        super().__init__(*args, **kwargs)

    def __str__(self):
        return '\n'.join((f'{x.source} -> {x.result or 'success'}' for x in self))