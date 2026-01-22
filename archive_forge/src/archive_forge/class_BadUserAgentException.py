import json
from typing import Any, Dict, List, Optional, Tuple, Type, Union
class BadUserAgentException(GithubException):
    """
    Exception raised when request is sent with a bad user agent header (when Github API replies with a 403 bad user agent HTML status)
    """