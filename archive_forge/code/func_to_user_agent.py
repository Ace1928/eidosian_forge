import platform
from typing import Union
from google.api_core import version as api_core_version
def to_user_agent(self):
    """Returns the user-agent string for this client info."""
    ua = ''
    if self.user_agent is not None:
        ua += '{user_agent} '
    ua += 'gl-python/{python_version} '
    if self.grpc_version is not None:
        ua += 'grpc/{grpc_version} '
    if self.rest_version is not None:
        ua += 'rest/{rest_version} '
    ua += 'gax/{api_core_version} '
    if self.gapic_version is not None:
        ua += 'gapic/{gapic_version} '
    if self.client_library_version is not None:
        ua += 'gccl/{client_library_version} '
    return ua.format(**self.__dict__).strip()