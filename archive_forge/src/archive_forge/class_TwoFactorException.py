import json
from typing import Any, Dict, List, Optional, Tuple, Type, Union
class TwoFactorException(GithubException):
    """
    Exception raised when Github requires a onetime password for two-factor authentication
    """