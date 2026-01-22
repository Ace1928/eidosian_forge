import functools
import re
from ovs.flow.decoders import decode_default
class KVDecoders(object):
    """KVDecoders class is used by KVParser to select how to decode the value
    of a specific keyword.

    A decoder is simply a function that accepts a value string and returns
    the value objects to be stored.
    The returned value may be of any type.

    Decoders may return a KeyValue instance to indicate that the keyword should
    also be modified to match the one provided in the returned KeyValue.

    The decoder to be used will be selected using the key as an index. If not
    found, the default decoder will be used. If free keys are found (i.e:
    keys without a value), the default_free decoder will be used. For that
    reason, the default_free decoder, must return both the key and value to be
    stored.

    Globally defined "strict" variable controls what to do when decoders do not
    contain a valid decoder for a key and a default function is not provided.
    If set to True (default), a ParseError is raised.
    If set to False, the value will be decoded as a string.

    Args:
        decoders (dict): Optional; A dictionary of decoders indexed by keyword.
        default (callable): Optional; A function to use if a match is not
            found in configured decoders. If not provided, the default behavior
            depends on "strict". The function must accept a the key and a value
            and return the decoded (key, value) tuple back.
        default_free (callable): Optional; The decoder used if a match is not
            found in configured decoders and it's a free value (e.g:
            a value without a key) Defaults to returning the free value as
            keyword and "True" as value.
            The callable must accept a string and return a key-value pair.
    """
    strict = True

    def __init__(self, decoders=None, default=None, default_free=None, ignore_case=False):
        if not decoders:
            self._decoders = dict()
        elif ignore_case:
            self._decoders = {k.lower(): v for k, v in decoders.items()}
        else:
            self._decoders = decoders
        self._default = default
        self._default_free = default_free or self._default_free_decoder
        self._ignore_case = ignore_case

    def decode(self, keyword, value_str):
        """Decode a keyword and value.

        Args:
            keyword (str): The keyword whose value is to be decoded.
            value_str (str): The value string.

        Returns:
            The key (str) and value(any) to be stored.
        """
        decoder = None
        if self._ignore_case:
            decoder = self._decoders.get(keyword.lower())
        else:
            decoder = self._decoders.get(keyword)
        if decoder:
            result = decoder(value_str)
            if isinstance(result, KeyValue):
                keyword = result.key
                value = result.value
            else:
                value = result
            return (keyword, value)
        else:
            if value_str:
                if self._default:
                    return self._default(keyword, value_str)
                if self.strict:
                    raise ParseError('Cannot parse key {}: No decoder found'.format(keyword))
                return (keyword, decode_default(value_str))
            return self._default_free(keyword)

    @staticmethod
    def _default_free_decoder(key):
        """Default decoder for free keywords."""
        return (key, True)