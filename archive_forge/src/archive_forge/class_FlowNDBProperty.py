import logging
from google.appengine.ext import ndb
from oauth2client import client
class FlowNDBProperty(ndb.PickleProperty):
    """App Engine NDB datastore Property for Flow.

    Serves the same purpose as the DB FlowProperty, but for NDB models.
    Since PickleProperty inherits from BlobProperty, the underlying
    representation of the data in the datastore will be the same as in the
    DB case.

    Utility property that allows easy storage and retrieval of an
    oauth2client.Flow
    """

    def _validate(self, value):
        """Validates a value as a proper Flow object.

        Args:
            value: A value to be set on the property.

        Raises:
            TypeError if the value is not an instance of Flow.
        """
        _LOGGER.info('validate: Got type %s', type(value))
        if value is not None and (not isinstance(value, client.Flow)):
            raise TypeError('Property {0} must be convertible to a flow instance; received: {1}.'.format(self._name, value))