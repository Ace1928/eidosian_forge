from oslo_utils import importutils
from manilaclient import api_versions
from manilaclient import exceptions
def _convert_to_api_version(version):
    """Convert version to an APIVersion object unless it already is one."""
    if hasattr(version, 'get_major_version'):
        api_version = version
    elif version in ('1', '1.0'):
        api_version = api_versions.APIVersion(api_versions.DEPRECATED_VERSION)
    elif version == '2':
        api_version = api_versions.APIVersion(api_versions.MIN_VERSION)
    else:
        api_version = api_versions.APIVersion(version)
    return api_version