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
class CsekKeyStore(object):
    """Represents a map from resource patterns to keys."""

    @classmethod
    def FromFile(cls, fname, allow_rsa_encrypted):
        """FromFile loads a CsekKeyStore from a file.

    Args:
      fname: str, the name of a file intended to contain a well-formed key file
      allow_rsa_encrypted: bool, whether to allow keys of type 'rsa-encrypted'

    Returns:
      A CsekKeyStore, if found

    Raises:
      googlecloudsdk.core.util.files.Error: If the file cannot be read or is
                                            larger than max_bytes.
    """
        content = console_io.ReadFromFileOrStdin(fname, binary=False)
        return cls(content, allow_rsa_encrypted)

    @staticmethod
    def FromArgs(args, allow_rsa_encrypted=False):
        """FromFile attempts to load a CsekKeyStore from a command's args.

    Args:
      args: CLI args with a csek_key_file field set
      allow_rsa_encrypted: bool, whether to allow keys of type 'rsa-encrypted'

    Returns:
      A CsekKeyStore, if a valid key file name is provided as csek_key_file
      None, if args.csek_key_file is None

    Raises:
      exceptions.BadFileException: there's a problem reading fname
      exceptions.InvalidKeyFileException: the key file failed to parse
        or was otherwise invalid
    """
        if args.csek_key_file is None:
            return None
        return CsekKeyStore.FromFile(args.csek_key_file, allow_rsa_encrypted)

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

    def __len__(self):
        return len(self.state)

    def LookupKey(self, resource, raise_if_missing=False):
        """Search for the unique key corresponding to a given resource.

    Args:
      resource: the resource to find a key for.
      raise_if_missing: bool, raise an exception if the resource is not found.

    Returns: CsekKeyBase, corresponding to the resource, or None if not found
      and not raise_if_missing.

    Raises:
      InvalidKeyFileException: if there are two records matching the resource.
      MissingCsekException: if raise_if_missing and no key is found
        for the provided resource.
    """
        assert isinstance(self.state, dict)
        search_state = (None, None)
        for pat, key in six.iteritems(self.state):
            if pat.Matches(resource):
                if search_state[0]:
                    raise InvalidKeyFileException('Uri patterns [{0}] and [{1}] both match resource [{2}].  Bailing out.'.format(search_state[0], pat, str(resource)))
                search_state = (pat, key)
        if raise_if_missing and search_state[1] is None:
            raise MissingCsekException(resource)
        return search_state[1]

    def __init__(self, json_string, allow_rsa_encrypted=False):
        self.state = CsekKeyStore._ParseAndValidate(json_string, allow_rsa_encrypted)