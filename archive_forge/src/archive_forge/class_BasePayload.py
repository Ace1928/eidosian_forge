import base64
import datetime
import struct
import uuid
from cryptography import fernet
import msgpack
from oslo_log import log
from oslo_utils import timeutils
from keystone.auth import plugins as auth_plugins
from keystone.common import fernet_utils as utils
from keystone.common import utils as ks_utils
import keystone.conf
from keystone import exception
from keystone.i18n import _
class BasePayload(object):
    version = None

    @classmethod
    def assemble(cls, user_id, methods, system, project_id, domain_id, expires_at, audit_ids, trust_id, federated_group_ids, identity_provider_id, protocol_id, access_token_id, app_cred_id, thumbprint):
        """Assemble the payload of a token.

        :param user_id: identifier of the user in the token request
        :param methods: list of authentication methods used
        :param system: a string including system scope information
        :param project_id: ID of the project to scope to
        :param domain_id: ID of the domain to scope to
        :param expires_at: datetime of the token's expiration
        :param audit_ids: list of the token's audit IDs
        :param trust_id: ID of the trust in effect
        :param federated_group_ids: list of group IDs from SAML assertion
        :param identity_provider_id: ID of the user's identity provider
        :param protocol_id: federated protocol used for authentication
        :param access_token_id: ID of the secret in OAuth1 authentication
        :param app_cred_id: ID of the application credential in effect
        :param thumbprint: thumbprint of the certificate in OAuth2 mTLS
        :returns: the payload of a token

        """
        raise NotImplementedError()

    @classmethod
    def disassemble(cls, payload):
        """Disassemble an unscoped payload into the component data.

        The tuple consists of::

            (user_id, methods, system, project_id, domain_id,
             expires_at_str, audit_ids, trust_id, federated_group_ids,
             identity_provider_id, protocol_id,` access_token_id, app_cred_id)

        * ``methods`` are the auth methods.

        Fields will be set to None if they didn't apply to this payload type.

        :param payload: this variant of payload
        :returns: a tuple of the payloads component data

        """
        raise NotImplementedError()

    @classmethod
    def convert_uuid_hex_to_bytes(cls, uuid_string):
        """Compress UUID formatted strings to bytes.

        :param uuid_string: uuid string to compress to bytes
        :returns: a byte representation of the uuid

        """
        uuid_obj = uuid.UUID(uuid_string)
        return uuid_obj.bytes

    @classmethod
    def convert_uuid_bytes_to_hex(cls, uuid_byte_string):
        """Generate uuid.hex format based on byte string.

        :param uuid_byte_string: uuid string to generate from
        :returns: uuid hex formatted string

        """
        uuid_obj = uuid.UUID(bytes=uuid_byte_string)
        return uuid_obj.hex

    @classmethod
    def _convert_time_string_to_float(cls, time_string):
        """Convert a time formatted string to a float.

        :param time_string: time formatted string
        :returns: a timestamp as a float

        """
        time_object = timeutils.parse_isotime(time_string)
        return (timeutils.normalize_time(time_object) - datetime.datetime.utcfromtimestamp(0)).total_seconds()

    @classmethod
    def _convert_float_to_time_string(cls, time_float):
        """Convert a floating point timestamp to a string.

        :param time_float: integer representing timestamp
        :returns: a time formatted strings

        """
        time_object = datetime.datetime.utcfromtimestamp(time_float)
        return ks_utils.isotime(time_object, subsecond=True)

    @classmethod
    def attempt_convert_uuid_hex_to_bytes(cls, value):
        """Attempt to convert value to bytes or return value.

        :param value: value to attempt to convert to bytes
        :returns: tuple containing boolean indicating whether user_id was
                  stored as bytes and uuid value as bytes or the original value

        """
        try:
            return (True, cls.convert_uuid_hex_to_bytes(value))
        except (ValueError, TypeError):
            return (False, value)

    @classmethod
    def base64_encode(cls, s):
        """Encode a URL-safe string.

        :type s: str
        :rtype: str

        """
        return base64.urlsafe_b64encode(s).decode('utf-8').rstrip('=')

    @classmethod
    def random_urlsafe_str_to_bytes(cls, s):
        """Convert string from :func:`random_urlsafe_str()` to bytes.

        :type s: str
        :rtype: bytes

        """
        s = str(s)
        return base64.urlsafe_b64decode(s + '==')

    @classmethod
    def _convert_or_decode(cls, is_stored_as_bytes, value):
        """Convert a value to text type, translating uuid -> hex if required.

        :param is_stored_as_bytes: whether value is already bytes
        :type is_stored_as_bytes: boolean
        :param value: value to attempt to convert to bytes
        :type value: str or bytes
        :rtype: str
        """
        if is_stored_as_bytes:
            return cls.convert_uuid_bytes_to_hex(value)
        elif isinstance(value, bytes):
            return value.decode('utf-8')
        return value