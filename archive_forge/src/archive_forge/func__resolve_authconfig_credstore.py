import base64
import json
import logging
import os
import platform
from typing import Any, Dict, Mapping, Optional, Tuple, Union
import dockerpycreds  # type: ignore
def _resolve_authconfig_credstore(self, registry: Optional[str], credstore_name: str) -> Optional[Dict[str, Any]]:
    if not registry or registry == INDEX_NAME:
        registry = INDEX_URL
    log.debug(f'Looking for auth entry for {repr(registry)}')
    store = self._get_store_instance(credstore_name)
    try:
        data = store.get(registry)
        res = {'ServerAddress': registry}
        if data['Username'] == TOKEN_USERNAME:
            res['IdentityToken'] = data['Secret']
        else:
            res.update({'Username': data['Username'], 'Password': data['Secret']})
        return res
    except (dockerpycreds.CredentialsNotFound, ValueError):
        log.debug('No entry found')
        return None
    except dockerpycreds.StoreError as e:
        raise DockerError(f'Credentials store error: {repr(e)}')