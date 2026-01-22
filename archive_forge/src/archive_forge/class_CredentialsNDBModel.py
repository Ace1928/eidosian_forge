import logging
from google.appengine.ext import ndb
from oauth2client import client
class CredentialsNDBModel(ndb.Model):
    """NDB Model for storage of OAuth 2.0 Credentials

    Since this model uses the same kind as CredentialsModel and has a
    property which can serialize and deserialize Credentials correctly, it
    can be used interchangeably with a CredentialsModel to access, insert
    and delete the same entities. This simply provides an NDB model for
    interacting with the same data the DB model interacts with.

    Storage of the model is keyed by the user.user_id().
    """
    credentials = CredentialsNDBProperty()

    @classmethod
    def _get_kind(cls):
        """Return the kind name for this class."""
        return 'CredentialsModel'