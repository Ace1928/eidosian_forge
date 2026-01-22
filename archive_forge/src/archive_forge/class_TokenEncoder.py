import base64
import json
import logging
from itertools import tee
import jmespath
from botocore.exceptions import PaginationError
from botocore.utils import merge_dicts, set_value_from_jmespath
class TokenEncoder:
    """Encodes dictionaries into opaque strings.

    This for the most part json dumps + base64 encoding, but also supports
    having bytes in the dictionary in addition to the types that json can
    handle by default.

    This is intended for use in encoding pagination tokens, which in some
    cases can be complex structures and / or contain bytes.
    """

    def encode(self, token):
        """Encodes a dictionary to an opaque string.

        :type token: dict
        :param token: A dictionary containing pagination information,
            particularly the service pagination token(s) but also other boto
            metadata.

        :rtype: str
        :returns: An opaque string
        """
        try:
            json_string = json.dumps(token)
        except (TypeError, UnicodeDecodeError):
            encoded_token, encoded_keys = self._encode(token, [])
            encoded_token['boto_encoded_keys'] = encoded_keys
            json_string = json.dumps(encoded_token)
        return base64.b64encode(json_string.encode('utf-8')).decode('utf-8')

    def _encode(self, data, path):
        """Encode bytes in given data, keeping track of the path traversed."""
        if isinstance(data, dict):
            return self._encode_dict(data, path)
        elif isinstance(data, list):
            return self._encode_list(data, path)
        elif isinstance(data, bytes):
            return self._encode_bytes(data, path)
        else:
            return (data, [])

    def _encode_list(self, data, path):
        """Encode any bytes in a list, noting the index of what is encoded."""
        new_data = []
        encoded = []
        for i, value in enumerate(data):
            new_path = path + [i]
            new_value, new_encoded = self._encode(value, new_path)
            new_data.append(new_value)
            encoded.extend(new_encoded)
        return (new_data, encoded)

    def _encode_dict(self, data, path):
        """Encode any bytes in a dict, noting the index of what is encoded."""
        new_data = {}
        encoded = []
        for key, value in data.items():
            new_path = path + [key]
            new_value, new_encoded = self._encode(value, new_path)
            new_data[key] = new_value
            encoded.extend(new_encoded)
        return (new_data, encoded)

    def _encode_bytes(self, data, path):
        """Base64 encode a byte string."""
        return (base64.b64encode(data).decode('utf-8'), [path])