import abc
import io
import json
import six
from google.auth import exceptions
@six.add_metaclass(abc.ABCMeta)
class FromServiceAccountMixin(object):
    """Mix-in to enable factory constructors for a Signer."""

    @abc.abstractmethod
    def from_string(cls, key, key_id=None):
        """Construct an Signer instance from a private key string.

        Args:
            key (str): Private key as a string.
            key_id (str): An optional key id used to identify the private key.

        Returns:
            google.auth.crypt.Signer: The constructed signer.

        Raises:
            ValueError: If the key cannot be parsed.
        """
        raise NotImplementedError('from_string must be implemented')

    @classmethod
    def from_service_account_info(cls, info):
        """Creates a Signer instance instance from a dictionary containing
        service account info in Google format.

        Args:
            info (Mapping[str, str]): The service account info in Google
                format.

        Returns:
            google.auth.crypt.Signer: The constructed signer.

        Raises:
            ValueError: If the info is not in the expected format.
        """
        if _JSON_FILE_PRIVATE_KEY not in info:
            raise exceptions.MalformedError('The private_key field was not found in the service account info.')
        return cls.from_string(info[_JSON_FILE_PRIVATE_KEY], info.get(_JSON_FILE_PRIVATE_KEY_ID))

    @classmethod
    def from_service_account_file(cls, filename):
        """Creates a Signer instance from a service account .json file
        in Google format.

        Args:
            filename (str): The path to the service account .json file.

        Returns:
            google.auth.crypt.Signer: The constructed signer.
        """
        with io.open(filename, 'r', encoding='utf-8') as json_file:
            data = json.load(json_file)
        return cls.from_service_account_info(data)