import errno
import json
import logging
import os
import threading
from oauth2client import client
from oauth2client import util
from oauth2client.contrib import locked_file
class _MultiStore(object):
    """A file backed store for multiple credentials."""

    @util.positional(2)
    def __init__(self, filename, warn_on_readonly=True):
        """Initialize the class.

        This will create the file if necessary.
        """
        self._file = locked_file.LockedFile(filename, 'r+', 'r')
        self._thread_lock = threading.Lock()
        self._read_only = False
        self._warn_on_readonly = warn_on_readonly
        self._create_file_if_needed()
        self._data = None

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

    def _create_file_if_needed(self):
        """Create an empty file if necessary.

        This method will not initialize the file. Instead it implements a
        simple version of "touch" to ensure the file has been created.
        """
        if not os.path.exists(self._file.filename()):
            old_umask = os.umask(127)
            try:
                open(self._file.filename(), 'a+b').close()
            finally:
                os.umask(old_umask)

    def _lock(self):
        """Lock the entire multistore."""
        self._thread_lock.acquire()
        try:
            self._file.open_and_lock()
        except (IOError, OSError) as e:
            if e.errno == errno.ENOSYS:
                logger.warn('File system does not support locking the credentials file.')
            elif e.errno == errno.ENOLCK:
                logger.warn('File system is out of resources for writing the credentials file (is your disk full?).')
            elif e.errno == errno.EDEADLK:
                logger.warn('Lock contention on multistore file, opening in read-only mode.')
            elif e.errno == errno.EACCES:
                logger.warn('Cannot access credentials file.')
            else:
                raise
        if not self._file.is_locked():
            self._read_only = True
            if self._warn_on_readonly:
                logger.warn('The credentials file (%s) is not writable. Opening in read-only mode. Any refreshed credentials will only be valid for this run.', self._file.filename())
        if os.path.getsize(self._file.filename()) == 0:
            logger.debug('Initializing empty multistore file')
            self._data = {}
            self._write()
        elif not self._read_only or self._data is None:
            self._refresh_data_cache()

    def _unlock(self):
        """Release the lock on the multistore."""
        self._file.unlock_and_close()
        self._thread_lock.release()

    def _locked_json_read(self):
        """Get the raw content of the multistore file.

        The multistore must be locked when this is called.

        Returns:
            The contents of the multistore decoded as JSON.
        """
        assert self._thread_lock.locked()
        self._file.file_handle().seek(0)
        return json.load(self._file.file_handle())

    def _locked_json_write(self, data):
        """Write a JSON serializable data structure to the multistore.

        The multistore must be locked when this is called.

        Args:
            data: The data to be serialized and written.
        """
        assert self._thread_lock.locked()
        if self._read_only:
            return
        self._file.file_handle().seek(0)
        json.dump(data, self._file.file_handle(), sort_keys=True, indent=2, separators=(',', ': '))
        self._file.file_handle().truncate()

    def _refresh_data_cache(self):
        """Refresh the contents of the multistore.

        The multistore must be locked when this is called.

        Raises:
            NewerCredentialStoreError: Raised when a newer client has written
            the store.
        """
        self._data = {}
        try:
            raw_data = self._locked_json_read()
        except Exception:
            logger.warn('Credential data store could not be loaded. Will ignore and overwrite.')
            return
        version = 0
        try:
            version = raw_data['file_version']
        except Exception:
            logger.warn('Missing version for credential data store. It may be corrupt or an old version. Overwriting.')
        if version > 1:
            raise NewerCredentialStoreError('Credential file has file_version of {0}. Only file_version of 1 is supported.'.format(version))
        credentials = []
        try:
            credentials = raw_data['data']
        except (TypeError, KeyError):
            pass
        for cred_entry in credentials:
            try:
                key, credential = self._decode_credential_from_json(cred_entry)
                self._data[key] = credential
            except:
                logger.info('Error decoding credential, skipping', exc_info=True)

    def _decode_credential_from_json(self, cred_entry):
        """Load a credential from our JSON serialization.

        Args:
            cred_entry: A dict entry from the data member of our format

        Returns:
            (key, cred) where the key is the key tuple and the cred is the
            OAuth2Credential object.
        """
        raw_key = cred_entry['key']
        key = _dict_to_tuple_key(raw_key)
        credential = None
        credential = client.Credentials.new_from_json(json.dumps(cred_entry['credential']))
        return (key, credential)

    def _write(self):
        """Write the cached data back out.

        The multistore must be locked.
        """
        raw_data = {'file_version': 1}
        raw_creds = []
        raw_data['data'] = raw_creds
        for cred_key, cred in self._data.items():
            raw_key = dict(cred_key)
            raw_cred = json.loads(cred.to_json())
            raw_creds.append({'key': raw_key, 'credential': raw_cred})
        self._locked_json_write(raw_data)

    def _get_all_credential_keys(self):
        """Gets all the registered credential keys in the multistore.

        Returns:
            A list of dictionaries corresponding to all the keys currently
            registered
        """
        return [dict(key) for key in self._data.keys()]

    def _get_credential(self, key):
        """Get a credential from the multistore.

        The multistore must be locked.

        Args:
            key: The key used to retrieve the credential

        Returns:
            The credential specified or None if not present
        """
        return self._data.get(key, None)

    def _update_credential(self, key, cred):
        """Update a credential and write the multistore.

        This must be called when the multistore is locked.

        Args:
            key: The key used to retrieve the credential
            cred: The OAuth2Credential to update/set
        """
        self._data[key] = cred
        self._write()

    def _delete_credential(self, key):
        """Delete a credential and write the multistore.

        This must be called when the multistore is locked.

        Args:
            key: The key used to retrieve the credential
        """
        try:
            del self._data[key]
        except KeyError:
            pass
        self._write()

    def _get_storage(self, key):
        """Get a Storage object to get/set a credential.

        This Storage is a 'view' into the multistore.

        Args:
            key: The key used to retrieve the credential

        Returns:
            A Storage object that can be used to get/set this cred
        """
        return self._Storage(self, key)