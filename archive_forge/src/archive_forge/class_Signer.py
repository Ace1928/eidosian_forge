import abc
import io
import json
import six
from google.auth import exceptions
@six.add_metaclass(abc.ABCMeta)
class Signer(object):
    """Abstract base class for cryptographic signers."""

    @abc.abstractproperty
    def key_id(self):
        """Optional[str]: The key ID used to identify this private key."""
        raise NotImplementedError('Key id must be implemented')

    @abc.abstractmethod
    def sign(self, message):
        """Signs a message.

        Args:
            message (Union[str, bytes]): The message to be signed.

        Returns:
            bytes: The signature of the message.
        """
        raise NotImplementedError('Sign must be implemented')