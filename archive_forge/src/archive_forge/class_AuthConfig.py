import base64
import json
import logging
import os
import platform
from typing import Any, Dict, Mapping, Optional, Tuple, Union
import dockerpycreds  # type: ignore
class AuthConfig(dict):

    def __init__(self, dct: Dict, credstore_env: Optional[Mapping]=None) -> None:
        super().__init__(dct)
        if 'auths' not in dct:
            dct['auths'] = {}
        self.update(dct)
        self._credstore_env = credstore_env
        self._stores: Dict[str, dockerpycreds.Store] = dict()

    @classmethod
    def parse_auth(cls, entries: Dict[str, Dict[str, Any]], raise_on_error: bool=False) -> Dict[str, Dict[str, Any]]:
        """Parse authentication entries.

        Arguments:
          entries:        Dict of authentication entries.
          raise_on_error: If set to true, an invalid format will raise
                          InvalidConfigFileError
        Returns:
          Authentication registry.
        """
        conf = {}
        for registry, entry in entries.items():
            if not isinstance(entry, dict):
                log.debug(f'Config entry for key {registry} is not auth config')
                if raise_on_error:
                    raise InvalidConfigFileError(f'Invalid configuration for registry {registry}')
                return {}
            if 'identitytoken' in entry:
                log.debug(f'Found an IdentityToken entry for registry {registry}')
                conf[registry] = {'IdentityToken': entry['identitytoken']}
                continue
            if 'auth' not in entry:
                log.debug(f'Auth data for {registry} is absent. Client might be using a credentials store instead.')
                conf[registry] = {}
                continue
            username, password = decode_auth(entry['auth'])
            log.debug(f'Found entry (registry={repr(registry)}, username={repr(username)})')
            conf[registry] = {'username': username, 'password': password, 'email': entry.get('email'), 'serveraddress': registry}
        return conf

    @classmethod
    def load_config(cls, config_path: Optional[str], config_dict: Optional[Dict[str, Any]], credstore_env: Optional[Mapping]=None) -> 'AuthConfig':
        """Load authentication data from a Docker configuration file.

        If the config_path is not passed in it looks for a configuration file in the
        root directory.

        Lookup priority:
            explicit config_path parameter > DOCKER_CONFIG environment
            variable > ~/.docker/config.json > ~/.dockercfg.
        """
        if not config_dict:
            config_file = find_config_file(config_path)
            if not config_file:
                return cls({}, credstore_env)
            try:
                with open(config_file) as f:
                    config_dict = json.load(f)
            except (OSError, KeyError, ValueError) as e:
                log.debug(e)
                return cls(_load_legacy_config(config_file), credstore_env)
        res = {}
        assert isinstance(config_dict, Dict)
        if config_dict.get('auths'):
            log.debug("Found 'auths' section")
            res.update({'auths': cls.parse_auth(config_dict.pop('auths'), raise_on_error=True)})
        if config_dict.get('credsStore'):
            log.debug("Found 'credsStore' section")
            res.update({'credsStore': config_dict.pop('credsStore')})
        if config_dict.get('credHelpers'):
            log.debug("Found 'credHelpers' section")
            res.update({'credHelpers': config_dict.pop('credHelpers')})
        if res:
            return cls(res, credstore_env)
        log.debug("Couldn't find auth-related section ; attempting to interpret as auth-only file")
        return cls({'auths': cls.parse_auth(config_dict)}, credstore_env)

    @property
    def auths(self) -> Dict[str, Dict[str, Any]]:
        return self.get('auths', {})

    @property
    def creds_store(self) -> Optional[str]:
        return self.get('credsStore', None)

    @property
    def cred_helpers(self) -> Dict:
        return self.get('credHelpers', {})

    @property
    def is_empty(self) -> bool:
        return not self.auths and (not self.creds_store) and (not self.cred_helpers)

    def resolve_authconfig(self, registry: Optional[str]=None) -> Optional[Dict[str, Any]]:
        """Return the authentication data for a specific registry.

        As with the Docker client, legacy entries in the config with full URLs are
        stripped down to hostnames before checking for a match. Returns None if no match
        was found.
        """
        if self.creds_store or self.cred_helpers:
            store_name = self.get_credential_store(registry)
            if store_name is not None:
                log.debug(f'Using credentials store {store_name!r}')
                cfg = self._resolve_authconfig_credstore(registry, store_name)
                if cfg is not None:
                    return cfg
                log.debug('No entry in credstore - fetching from auth dict')
        registry = resolve_index_name(registry) if registry else INDEX_NAME
        log.debug(f'Looking for auth entry for {repr(registry)}')
        if registry in self.auths:
            log.debug(f'Found {repr(registry)}')
            return self.auths[registry]
        for key, conf in self.auths.items():
            if resolve_index_name(key) == registry:
                log.debug(f'Found {repr(key)}')
                return conf
        log.debug('No entry found')
        return None

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

    def _get_store_instance(self, name: str) -> 'dockerpycreds.Store':
        if name not in self._stores:
            self._stores[name] = dockerpycreds.Store(name, environment=self._credstore_env)
        return self._stores[name]

    def get_credential_store(self, registry: Optional[str]) -> Optional[str]:
        if not registry or registry == INDEX_NAME:
            registry = INDEX_URL
        return self.cred_helpers.get(registry) or self.creds_store

    def get_all_credentials(self) -> Dict[str, Dict[str, Any]]:
        auth_data = self.auths.copy()
        if self.creds_store:
            store = self._get_store_instance(self.creds_store)
            for k in store.list().keys():
                auth_data[k] = self._resolve_authconfig_credstore(k, self.creds_store)
        for reg, store_name in self.cred_helpers.items():
            auth_data[reg] = self._resolve_authconfig_credstore(reg, store_name)
        return auth_data

    def add_auth(self, reg: str, data: Dict[str, Any]) -> None:
        self['auths'][reg] = data