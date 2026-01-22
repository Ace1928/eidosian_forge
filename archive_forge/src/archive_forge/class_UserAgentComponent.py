import os
import platform
from copy import copy
from string import ascii_letters, digits
from typing import NamedTuple, Optional
from botocore import __version__ as botocore_version
from botocore.compat import HAS_CRT
class UserAgentComponent(NamedTuple):
    """
    Component of a Botocore User-Agent header string in the standard format.

    Each component consists of a prefix, a name, and a value. In the string
    representation these are combined in the format ``prefix/name#value``.

    This class is considered private and is subject to abrupt breaking changes.
    """
    prefix: str
    name: str
    value: Optional[str] = None

    def to_string(self):
        """Create string like 'prefix/name#value' from a UserAgentComponent."""
        clean_prefix = sanitize_user_agent_string_component(self.prefix, allow_hash=True)
        clean_name = sanitize_user_agent_string_component(self.name, allow_hash=False)
        if self.value is None or self.value == '':
            return f'{clean_prefix}/{clean_name}'
        clean_value = sanitize_user_agent_string_component(self.value, allow_hash=True)
        return f'{clean_prefix}/{clean_name}#{clean_value}'