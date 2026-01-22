import filecmp
import logging
import os
import requests
import wandb
def _check_entry_is_downloable(entry):
    url = entry._download_url
    expected_size = entry.size
    try:
        resp = requests.head(url, allow_redirects=True)
    except Exception as e:
        logger.error(f'Problem validating entry={entry!r}, e={e!r}')
        return False
    if resp.status_code != 200:
        return False
    actual_size = resp.headers.get('content-length', -1)
    actual_size = int(actual_size)
    if expected_size == actual_size:
        return True
    return False