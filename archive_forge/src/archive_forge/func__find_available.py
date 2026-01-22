from typing import Dict, Optional, Tuple
import requests
import wandb
def _find_available(current_version: str) -> Optional[Tuple[str, bool, bool, bool, Optional[str]]]:
    from wandb.util import parse_version
    pypi_url = f'https://pypi.org/pypi/{wandb._wandb_module}/json'
    yanked_dict = {}
    try:
        async_requests_get = wandb.util.async_call(requests.get, timeout=5)
        data, thread = async_requests_get(pypi_url, timeout=3)
        if not data or isinstance(data, Exception):
            return None
        data = data.json()
        latest_version = data['info']['version']
        release_list = data['releases'].keys()
        for version, fields in data['releases'].items():
            for item in fields:
                yanked = item.get('yanked')
                yanked_reason = item.get('yanked_reason')
                if yanked:
                    yanked_dict[version] = yanked_reason
    except Exception:
        return None
    pip_prerelease = False
    deleted = False
    yanked = False
    yanked_reason = None
    parsed_current_version = parse_version(current_version)
    if current_version in release_list:
        yanked = current_version in yanked_dict
        yanked_reason = yanked_dict.get(current_version)
    else:
        deleted = True
    if parse_version(latest_version) <= parsed_current_version:
        if not parsed_current_version.is_prerelease:
            return None
        release_list = map(parse_version, release_list)
        release_list = filter(lambda v: v.is_prerelease, release_list)
        release_list = filter(lambda v: v.base_version == parsed_current_version.base_version, release_list)
        release_list = sorted(release_list)
        if not release_list:
            return None
        parsed_latest_version = release_list[-1]
        if parsed_latest_version <= parsed_current_version:
            return None
        latest_version = str(parsed_latest_version)
        pip_prerelease = True
    return (latest_version, pip_prerelease, deleted, yanked, yanked_reason)