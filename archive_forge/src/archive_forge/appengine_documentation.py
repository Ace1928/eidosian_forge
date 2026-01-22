import requests
import warnings
from requests import adapters
from requests import sessions
from .. import exceptions as exc
from .._compat import gaecontrib
from .._compat import timeout
Sets up all Sessions to use AppEngineAdapter by default.

    If you don't want to deal with configuring your own Sessions,
    or if you use libraries that use requests directly (ie requests.post),
    then you may prefer to monkeypatch and auto-configure all Sessions.

    .. warning: :

        If ``validate_certificate`` is ``False``, certification validation will
        effectively be disabled for all requests.
    