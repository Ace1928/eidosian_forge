import errno
import json
import logging
import os
import threading
from oauth2client import client
from oauth2client import util
from oauth2client.contrib import locked_file
class _Storage(client.Storage):
    """A Storage object that can read/write a single credential."""

    def __init__(self, multistore, key):
        self._multistore = multistore
        self._key = key

    def acquire_lock(self):
        """Acquires any lock necessary to access this Storage.

            This lock is not reentrant.
            """
        self._multistore._lock()

    def release_lock(self):
        """Release the Storage lock.

            Trying to release a lock that isn't held will result in a
            RuntimeError.
            """
        self._multistore._unlock()

    def locked_get(self):
        """Retrieve credential.

            The Storage lock must be held when this is called.

            Returns:
                oauth2client.client.Credentials
            """
        credential = self._multistore._get_credential(self._key)
        if credential:
            credential.set_store(self)
        return credential

    def locked_put(self, credentials):
        """Write a credential.

            The Storage lock must be held when this is called.

            Args:
                credentials: Credentials, the credentials to store.
            """
        self._multistore._update_credential(self._key, credentials)

    def locked_delete(self):
        """Delete a credential.

            The Storage lock must be held when this is called.

            Args:
                credentials: Credentials, the credentials to store.
            """
        self._multistore._delete_credential(self._key)