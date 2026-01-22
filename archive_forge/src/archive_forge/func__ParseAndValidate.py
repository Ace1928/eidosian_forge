from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import abc
import base64
import json
import re
from googlecloudsdk.api_lib.compute import exceptions
from googlecloudsdk.core import resources
from googlecloudsdk.core.console import console_io
import six
@staticmethod
def _ParseAndValidate(s, allow_rsa_encrypted=False):
    """_ParseAndValidate(s) inteprets s as a csek key file.

    Args:
      s: str, an input to parse
      allow_rsa_encrypted: bool, whether to allow RSA-wrapped keys

    Returns:
      a valid state object

    Raises:
      InvalidKeyFileException: if the input doesn't parse or is not well-formed.
    """
    assert isinstance(s, six.string_types)
    state = {}
    try:
        records = json.loads(s)
        if not isinstance(records, list):
            raise InvalidKeyFileException("Key file's top-level element must be a JSON list.")
        for key_record in records:
            if not isinstance(key_record, dict):
                raise InvalidKeyFileException('Key file records must be JSON objects, but [{0}] found.'.format(json.dumps(key_record)))
            if set(key_record.keys()) != EXPECTED_RECORD_KEY_KEYS:
                raise InvalidKeyFileException('Record [{0}] has incorrect json keys; [{1}] expected'.format(json.dumps(key_record), ','.join(EXPECTED_RECORD_KEY_KEYS)))
            pattern = UriPattern(key_record['uri'])
            try:
                state[pattern] = CsekKeyBase.MakeKey(key_material=key_record['key'], key_type=key_record['key-type'], allow_rsa_encrypted=allow_rsa_encrypted)
            except InvalidKeyExceptionNoContext as e:
                raise InvalidKeyException(key=e.key, key_id=pattern, issue=e.issue)
    except ValueError as e:
        raise InvalidKeyFileException(*e.args)
    assert isinstance(state, dict)
    return state