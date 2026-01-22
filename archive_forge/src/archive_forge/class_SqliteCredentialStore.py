from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import abc
import base64
import collections
import copy
import datetime
import enum
import hashlib
import json
import os
import sqlite3
from google.auth import compute_engine as google_auth_compute_engine
from google.auth import credentials as google_auth_creds
from google.auth import exceptions as google_auth_exceptions
from google.auth import external_account as google_auth_external_account
from google.auth import external_account_authorized_user as google_auth_external_account_authorized_user
from google.auth import impersonated_credentials as google_auth_impersonated
from googlecloudsdk.core import config
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core.credentials import devshell as c_devshell
from googlecloudsdk.core.credentials import exceptions as c_exceptions
from googlecloudsdk.core.credentials import introspect as c_introspect
from googlecloudsdk.core.util import files
from oauth2client import client
from oauth2client import service_account
from oauth2client.contrib import gce as oauth2client_gce
import six
class SqliteCredentialStore(CredentialStore):
    """Sqllite backed credential store."""

    def __init__(self, store_file):
        self._cursor = _SqlCursor(store_file)
        self._Execute('CREATE TABLE IF NOT EXISTS "{}" (account_id TEXT PRIMARY KEY, value BLOB)'.format(_CREDENTIAL_TABLE_NAME))
        config_store = config.GetConfigStore()
        if not config_store.Get('cred_token_store_formatted'):
            self.FormatAccountIdColumn()

    def _Execute(self, *args):
        with self._cursor as cur:
            return cur.Execute(*args)

    def FormatAccountIdColumn(self):
        """Format the account id column.

    Before we introduce the formatted account id concept, the existing table
    uses the account id value as the key. Therefore we need to load the table
    and replace these account ids with formatted account ids.
    """
        with self._cursor as cur:
            table = cur.Execute('SELECT account_id, value FROM "{}"'.format(_CREDENTIAL_TABLE_NAME)).fetchall()
            for row in table:
                account_id, cred_json = (row[0], row[1])
                if '#' not in account_id:
                    creds = FromJsonGoogleAuth(cred_json)
                    formatted_account_id = _AccountIdFormatter.GetFormattedAccountId(account_id, creds)
                    if account_id != formatted_account_id:
                        cur.Execute('DELETE FROM "{}" WHERE account_id = ?'.format(_CREDENTIAL_TABLE_NAME), (account_id,))
                        cur.Execute('INSERT INTO "{}" (account_id, value) VALUES (?,?)'.format(_CREDENTIAL_TABLE_NAME), (formatted_account_id, cred_json))
            config_store = config.GetConfigStore()
            config_store.Set('cred_token_store_formatted', True)

    def GetAccounts(self):
        """Get all accounts.

    Returns:
      set[str], A set of account ids.
    """
        with self._cursor as cur:
            accounts = set()
            for formatted_account_id, in cur.Execute('SELECT account_id FROM "{}" ORDER BY rowid'.format(_CREDENTIAL_TABLE_NAME)):
                account_id, _ = _AccountIdFormatter.GetAccountIdAndUniverseDomain(formatted_account_id)
                accounts.add(account_id)
        return accounts

    def GetAccountsWithUniverseDomain(self):
        """Get all accounts and their corresponding universe domains.

    Returns:
      collections.defaultdict, A dictionary where the key is the account_id and
        the value is the universe domain list.
    """
        accounts_dict = collections.defaultdict(list)
        with self._cursor as cur:
            for formatted_account_id in cur.Execute('SELECT account_id FROM "{}" ORDER BY rowid'.format(_CREDENTIAL_TABLE_NAME)):
                account_id, universe_domain = _AccountIdFormatter.GetAccountIdAndUniverseDomain(formatted_account_id[0])
                accounts_dict[account_id].append(universe_domain)
        return accounts_dict

    def Load(self, account_id, use_google_auth=True):
        """Load the credentials for the account_id.

    Args:
      account_id: str, The account_id of the credential to load.
      use_google_auth: bool, Whether google-auth lib should be used. Default is
        True.

    Returns:
      google.auth.credentials.Credentials or client.OAuth2Credentials, The
        loaded credentials.

    Raises:
      googlecloudsdk.core.credentials.creds.InvalidCredentialsError: If problem
        happens when loading credentials.
    """
        if not use_google_auth:
            with self._cursor as cur:
                cred_json = cur.Execute('SELECT value FROM "{}" WHERE account_id = ?'.format(_CREDENTIAL_TABLE_NAME), (account_id,)).fetchone()
            if cred_json is None:
                return None
            return FromJson(cred_json[0])
        with self._cursor as cur:
            table = cur.Execute('SELECT account_id, value FROM "{}" WHERE account_id = ? OR account_id LIKE ?'.format(_CREDENTIAL_TABLE_NAME), (account_id, account_id + '#%')).fetchall()
        if not table:
            return None
        universe_domain_property = properties.VALUES.core.universe_domain
        universe_domains = []
        creds = None
        for formatted_account_id, cred_json in table:
            _, universe_domain = _AccountIdFormatter.GetAccountIdAndUniverseDomain(formatted_account_id)
            universe_domains.append(universe_domain)
            if universe_domain == universe_domain_property.Get():
                creds = FromJsonGoogleAuth(cred_json)
        if not creds:
            raise InvalidCredentialsError('The account [{account_id}] is available in the following universe domain(s): [{universe_domains}], but it is not available in [{universe_property}] which is specified by the [core/universe_domain] property. Update your active account to an account from {universe_property} or update the [core/universe_domain] property to one of [{universe_domains}].'.format(account_id=account_id, universe_property=universe_domain_property.Get(), universe_domains=', '.join(universe_domains)))
        return creds

    def Store(self, account_id, credentials):
        """Stores the input credentials to the record of account_id in the cache.

    Args:
      account_id: string, the account ID of the input credentials.
      credentials: google.auth.credentials.Credentials or
        client.OAuth2Credentials, the credentials to be stored.
    """
        if IsOauth2ClientCredentials(credentials):
            value = ToJson(credentials)
            self._Execute('REPLACE INTO "{}" (account_id, value) VALUES (?,?)'.format(_CREDENTIAL_TABLE_NAME), (account_id, value))
        else:
            value = ToJsonGoogleAuth(credentials)
            formatted_account_id = _AccountIdFormatter.GetFormattedAccountId(account_id, credentials)
            self._Execute('REPLACE INTO "{}" (account_id, value) VALUES (?,?)'.format(_CREDENTIAL_TABLE_NAME), (formatted_account_id, value))

    def Remove(self, account_id):
        formatted_account_id = _AccountIdFormatter.GetFormattedAccountId(account_id, None)
        self._Execute('DELETE FROM "{}" WHERE account_id = ?'.format(_CREDENTIAL_TABLE_NAME), (formatted_account_id,))