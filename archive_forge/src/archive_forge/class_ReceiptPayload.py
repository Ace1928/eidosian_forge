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
class ReceiptPayload(object):

    @classmethod
    def assemble(cls, user_id, methods, expires_at):
        """Assemble the payload of a receipt.

        :param user_id: identifier of the user in the receipt request
        :param methods: list of authentication methods used
        :param expires_at: datetime of the receipt's expiration
        :returns: the payload of a receipt

        """
        b_user_id = cls.attempt_convert_uuid_hex_to_bytes(user_id)
        methods = auth_plugins.convert_method_list_to_integer(methods)
        expires_at_int = cls._convert_time_string_to_float(expires_at)
        return (b_user_id, methods, expires_at_int)

    @classmethod
    def disassemble(cls, payload):
        """Disassemble a payload into the component data.

        The tuple consists of::

            (user_id, methods, expires_at_str)

        * ``methods`` are the auth methods.

        :param payload: this variant of payload
        :returns: a tuple of the payloads component data

        """
        is_stored_as_bytes, user_id = payload[0]
        if is_stored_as_bytes:
            user_id = cls.convert_uuid_bytes_to_hex(user_id)
        methods = auth_plugins.convert_integer_to_method_list(payload[1])
        expires_at_str = cls._convert_float_to_time_string(payload[2])
        return (user_id, methods, expires_at_str)

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
        except ValueError:
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