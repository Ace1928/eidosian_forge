import logging
from google.appengine.ext import ndb
from oauth2client import client
class CredentialsNDBProperty(ndb.BlobProperty):
    """App Engine NDB datastore Property for Credentials.

    Serves the same purpose as the DB CredentialsProperty, but for NDB
    models. Since CredentialsProperty stores data as a blob and this
    inherits from BlobProperty, the data in the datastore will be the same
    as in the DB case.

    Utility property that allows easy storage and retrieval of Credentials
    and subclasses.
    """

    def _validate(self, value):
        """Validates a value as a proper credentials object.

        Args:
            value: A value to be set on the property.

        Raises:
            TypeError if the value is not an instance of Credentials.
        """
        _LOGGER.info('validate: Got type %s', type(value))
        if value is not None and (not isinstance(value, client.Credentials)):
            raise TypeError('Property {0} must be convertible to a credentials instance; received: {1}.'.format(self._name, value))

    def _to_base_type(self, value):
        """Converts our validated value to a JSON serialized string.

        Args:
            value: A value to be set in the datastore.

        Returns:
            A JSON serialized version of the credential, else '' if value
            is None.
        """
        if value is None:
            return ''
        else:
            return value.to_json()

    def _from_base_type(self, value):
        """Converts our stored JSON string back to the desired type.

        Args:
            value: A value from the datastore to be converted to the
                   desired type.

        Returns:
            A deserialized Credentials (or subclass) object, else None if
            the value can't be parsed.
        """
        if not value:
            return None
        try:
            credentials = client.Credentials.new_from_json(value)
        except ValueError:
            credentials = None
        return credentials