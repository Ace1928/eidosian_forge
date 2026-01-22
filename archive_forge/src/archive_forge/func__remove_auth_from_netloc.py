import json
import re
import urllib.parse
from typing import Any, Dict, Iterable, Optional, Type, TypeVar, Union
def _remove_auth_from_netloc(self, netloc: str) -> str:
    if '@' not in netloc:
        return netloc
    user_pass, netloc_no_user_pass = netloc.split('@', 1)
    if isinstance(self.info, VcsInfo) and self.info.vcs == 'git' and (user_pass == 'git'):
        return netloc
    if ENV_VAR_RE.match(user_pass):
        return netloc
    return netloc_no_user_pass