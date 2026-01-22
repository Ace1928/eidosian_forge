import os
import platform
from copy import copy
from string import ascii_letters, digits
from typing import NamedTuple, Optional
from botocore import __version__ as botocore_version
from botocore.compat import HAS_CRT
def _build_config_metadata(self):
    """
        Build the configuration components of the User-Agent header string.

        Returns a list of components with prefix "cfg" followed by the config
        setting name and its value. Tracked configuration settings may be
        added or removed in future versions.
        """
    if not self._client_config or not self._client_config.retries:
        return []
    retry_mode = self._client_config.retries.get('mode')
    cfg_md = [UserAgentComponent('cfg', 'retry-mode', retry_mode)]
    if self._client_config.endpoint_discovery_enabled:
        cfg_md.append(UserAgentComponent('cfg', 'endpoint-discovery'))
    return cfg_md