import cgi
import json
import logging
import os
import pickle
import threading
from google.appengine.api import app_identity
from google.appengine.api import memcache
from google.appengine.api import users
from google.appengine.ext import db
from google.appengine.ext.webapp.util import login_required
import webapp2 as webapp
import oauth2client
from oauth2client import _helpers
from oauth2client import client
from oauth2client import clientsecrets
from oauth2client import transport
from oauth2client.contrib import xsrfutil
class StorageByKeyName(client.Storage):
    """Store and retrieve a credential to and from the App Engine datastore.

    This Storage helper presumes the Credentials have been stored as a
    CredentialsProperty or CredentialsNDBProperty on a datastore model class,
    and that entities are stored by key_name.
    """

    @_helpers.positional(4)
    def __init__(self, model, key_name, property_name, cache=None, user=None):
        """Constructor for Storage.

        Args:
            model: db.Model or ndb.Model, model class
            key_name: string, key name for the entity that has the credentials
            property_name: string, name of the property that is a
                           CredentialsProperty or CredentialsNDBProperty.
            cache: memcache, a write-through cache to put in front of the
                   datastore. If the model you are using is an NDB model, using
                   a cache will be redundant since the model uses an instance
                   cache and memcache for you.
            user: users.User object, optional. Can be used to grab user ID as a
                  key_name if no key name is specified.
        """
        super(StorageByKeyName, self).__init__()
        if key_name is None:
            if user is None:
                raise ValueError('StorageByKeyName called with no key name or user.')
            key_name = user.user_id()
        self._model = model
        self._key_name = key_name
        self._property_name = property_name
        self._cache = cache

    def _is_ndb(self):
        """Determine whether the model of the instance is an NDB model.

        Returns:
            Boolean indicating whether or not the model is an NDB or DB model.
        """
        if isinstance(self._model, type):
            if _NDB_MODEL is not None and issubclass(self._model, _NDB_MODEL):
                return True
            elif issubclass(self._model, db.Model):
                return False
        raise TypeError('Model class not an NDB or DB model: {0}.'.format(self._model))

    def _get_entity(self):
        """Retrieve entity from datastore.

        Uses a different model method for db or ndb models.

        Returns:
            Instance of the model corresponding to the current storage object
            and stored using the key name of the storage object.
        """
        if self._is_ndb():
            return self._model.get_by_id(self._key_name)
        else:
            return self._model.get_by_key_name(self._key_name)

    def _delete_entity(self):
        """Delete entity from datastore.

        Attempts to delete using the key_name stored on the object, whether or
        not the given key is in the datastore.
        """
        if self._is_ndb():
            _NDB_KEY(self._model, self._key_name).delete()
        else:
            entity_key = db.Key.from_path(self._model.kind(), self._key_name)
            db.delete(entity_key)

    @db.non_transactional(allow_existing=True)
    def locked_get(self):
        """Retrieve Credential from datastore.

        Returns:
            oauth2client.Credentials
        """
        credentials = None
        if self._cache:
            json = self._cache.get(self._key_name)
            if json:
                credentials = client.Credentials.new_from_json(json)
        if credentials is None:
            entity = self._get_entity()
            if entity is not None:
                credentials = getattr(entity, self._property_name)
                if self._cache:
                    self._cache.set(self._key_name, credentials.to_json())
        if credentials and hasattr(credentials, 'set_store'):
            credentials.set_store(self)
        return credentials

    @db.non_transactional(allow_existing=True)
    def locked_put(self, credentials):
        """Write a Credentials to the datastore.

        Args:
            credentials: Credentials, the credentials to store.
        """
        entity = self._model.get_or_insert(self._key_name)
        setattr(entity, self._property_name, credentials)
        entity.put()
        if self._cache:
            self._cache.set(self._key_name, credentials.to_json())

    @db.non_transactional(allow_existing=True)
    def locked_delete(self):
        """Delete Credential from datastore."""
        if self._cache:
            self._cache.delete(self._key_name)
        self._delete_entity()