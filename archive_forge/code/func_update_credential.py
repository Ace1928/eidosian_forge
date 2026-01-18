import abc
from oslo_log import log
from keystone import exception
@abc.abstractmethod
def update_credential(self, credential_id, credential):
    """Update an existing credential.

        :raises keystone.exception.CredentialNotFound: If credential doesn't
            exist.
        :raises keystone.exception.Conflict: If a duplicate credential exists.

        """
    raise exception.NotImplemented()