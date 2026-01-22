from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import json
import logging
import os
import sqlite3
import time
from typing import Dict
import uuid
import googlecloudsdk
from googlecloudsdk.core import exceptions
from googlecloudsdk.core.configurations import named_configs
from googlecloudsdk.core.util import encoding
from googlecloudsdk.core.util import files as file_utils
from googlecloudsdk.core.util import pkg_resources
from googlecloudsdk.core.util import platforms
import six
class SqliteConfigStore(object):
    """Sqlite backed config store."""

    def __init__(self, store_file, config_name):
        self._cursor = _SqlCursor(store_file)
        self._config_name = config_name
        self._Execute('CREATE TABLE IF NOT EXISTS config (config_attr TEXT PRIMARY KEY, value BLOB)')

    def _Execute(self, *args):
        with self._cursor as cur:
            return cur.Execute(*args)

    def _LoadAttribute(self, config_attr, required):
        """Returns the attribute value from the SQLite table."""
        loaded_config = None
        with self._cursor as cur:
            try:
                loaded_config = cur.Execute('SELECT value FROM config WHERE config_attr = ?', (config_attr,)).fetchone()
            except sqlite3.OperationalError as e:
                logging.warning('Could not load config attribute [%s] in cache: %s', config_attr, str(e))
        if loaded_config is None and required:
            logging.warning('The required config attribute [%s] is not set.', config_attr)
        elif loaded_config is None:
            return None
        return loaded_config[0]

    def _Load(self):
        """Returns the entire config object from the SQLite table."""
        loaded_config = None
        with self._cursor as cur:
            try:
                loaded_config = cur.Execute('SELECT config_attr, value FROM config ORDER BY rowid').fetchall()
            except sqlite3.OperationalError as e:
                logging.warning('Could not store config attribute in cache: %s', str(e))
        return loaded_config

    def Get(self, config_attr, required=False):
        """Gets the given attribute.

    Args:
      config_attr: string, The attribute key to get.
      required: bool, True to raise an exception if the attribute is not set.

    Returns:
      object, The value of the attribute, or None if it is not set.
    """
        attr_value = self._LoadAttribute(config_attr, required)
        if attr_value is None or Stringize(attr_value).lower() == 'none':
            return None
        return attr_value

    def Set(self, config_attr, config_value):
        """Sets the value for an attribute.

    Args:
      config_attr: string, the primary key of the attribute to store.
      config_value: obj, the value of the config key attribute.
    """
        if isinstance(config_value, Dict):
            config_value = json.dumps(config_value).encode('utf-8')
        self._StoreAttribute(config_attr, config_value)

    def _GetBoolAttribute(self, config_attr, required, validate=True):
        """Gets the given attribute in bool form.

    Args:
      config_attr: string, The attribute key to get.
      required: bool, True to raise an exception if the attribute is not set.
      validate: bool, True to validate the value

    Returns:
      bool, The value of the attribute, or None if it is not set.
    """
        attr_value = self._LoadAttribute(config_attr, required)
        if validate:
            _BooleanValidator(config_attr, attr_value)
        if attr_value is None:
            return None
        attr_string_value = Stringize(attr_value).lower()
        if attr_string_value == 'none':
            return None
        return attr_string_value in ['1', 'true', 'on', 'yes', 'y']

    def GetBool(self, config_attr, required=False, validate=True):
        """Gets the boolean value for this attribute.

    Args:
      config_attr: string, The attribute key to get.
      required: bool, True to raise an exception if the attribute is not set.
      validate: bool, Whether or not to run the fetched value through the
        validation function.

    Returns:
      bool, The boolean value for this attribute, or None if it is not set.

    Raises:
      InvalidValueError: if value is not boolean
    """
        value = self._GetBoolAttribute(config_attr, required, validate=validate)
        return value

    def _GetIntAttribute(self, config_attr, required):
        """Gets the given attribute in integer form.

    Args:
      config_attr: string, The attribute key to get.
      required: bool, True to raise an exception if the attribute is not set.

    Returns:
      int, The integer value of the attribute, or None if it is not set.
    """
        attr_value = self._LoadAttribute(config_attr, required)
        if attr_value is None:
            return None
        try:
            return int(attr_value)
        except ValueError:
            raise InvalidValueError('The attribute [{attr}] must have an integer value: [{value}]'.format(attr=config_attr, value=attr_value))

    def GetInt(self, config_attr, required=False):
        """Gets the integer value for this attribute.

    Args:
      config_attr: string, The attribute key to get.
      required: bool, True to raise an exception if the attribute is not set.

    Returns:
      int, The integer value for this attribute.
    """
        value = self._GetIntAttribute(config_attr, required)
        return value

    def GetJSON(self, config_attr, required=False):
        """Gets the JSON value for this attribute.

    Args:
      config_attr: string, The attribute key to get.
      required: bool, True to raise an exception if the attribute is not set.

    Returns:
      The JSON value for this attribute or None.
    """
        attr_value = self._LoadAttribute(config_attr, required)
        if attr_value is None:
            return None
        try:
            return json.loads(attr_value)
        except ValueError:
            return attr_value

    def _StoreAttribute(self, config_attr: str, config_value):
        """Stores the input config attributes to the record of config_name in the cache.

    Args:
      config_attr: string, the primary key of the attribute to store.
      config_value: obj, the value of the config key attribute.
    """
        self._Execute('REPLACE INTO config (config_attr, value) VALUES (?,?)', (config_attr, config_value))

    def DeleteConfig(self):
        """Permanently erases the config .db file."""
        config_db_path = Paths().config_db_path.format(self._config_name)
        try:
            if os.path.exists(config_db_path):
                os.remove(config_db_path)
            else:
                logging.warning('Failed to delete config DB: path [%s] does not exist.', config_db_path)
        except OSError as e:
            logging.warning('Could not delete config from cache: %s', str(e))

    def _DeleteAttribute(self, config_attr: str):
        """Deletes a specified attribute from the config."""
        try:
            self._Execute('DELETE FROM config WHERE config_attr = ?', (config_attr,))
            with self._cursor as cur:
                if cur.RowCount() < 1:
                    logging.warning('Could not delete attribute [%s] from cache in config store [%s].', config_attr, self._config_name)
        except sqlite3.OperationalError as e:
            logging.warning('Could not delete attribute [%s] from cache: %s', config_attr, str(e))

    def Remove(self, config_attr):
        """Removes an attribute from the config."""
        self._DeleteAttribute(config_attr)