import os
import platform
from copy import copy
from string import ascii_letters, digits
from typing import NamedTuple, Optional
from botocore import __version__ as botocore_version
from botocore.compat import HAS_CRT
def _build_sdk_metadata(self):
    """
        Build the SDK name and version component of the User-Agent header.

        For backwards-compatibility both session-level and client-level config
        of custom tool names are honored. If this removes the Botocore
        information from the start of the string, Botocore's name and version
        are included as a separate field with "md" prefix.
        """
    sdk_md = []
    if self._session_user_agent_name and self._session_user_agent_version and (self._session_user_agent_name != _USERAGENT_SDK_NAME or self._session_user_agent_version != botocore_version):
        sdk_md.extend([UserAgentComponent(self._session_user_agent_name, self._session_user_agent_version), UserAgentComponent('md', _USERAGENT_SDK_NAME, botocore_version)])
    else:
        sdk_md.append(UserAgentComponent(_USERAGENT_SDK_NAME, botocore_version))
    if self._crt_version is not None:
        sdk_md.append(UserAgentComponent('md', 'awscrt', self._crt_version))
    return sdk_md