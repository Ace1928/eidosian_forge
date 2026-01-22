import io
import os
import sys
from pyasn1 import debug
from pyasn1 import error
from pyasn1.codec.ber import eoo
from pyasn1.codec.streaming import asSeekableStream
from pyasn1.codec.streaming import isEndOfStream
from pyasn1.codec.streaming import peekIntoStream
from pyasn1.codec.streaming import readFromStream
from pyasn1.compat import _MISSING
from pyasn1.compat.integer import from_bytes
from pyasn1.compat.octets import oct2int, octs2ints, ints2octs, null
from pyasn1.error import PyAsn1Error
from pyasn1.type import base
from pyasn1.type import char
from pyasn1.type import tag
from pyasn1.type import tagmap
from pyasn1.type import univ
from pyasn1.type import useful
class StreamingDecoder(object):
    """Create an iterator that turns BER/CER/DER byte stream into ASN.1 objects.

    On each iteration, consume whatever BER/CER/DER serialization is
    available in the `substrate` stream-like object and turns it into
    one or more, possibly nested, ASN.1 objects.

    Parameters
    ----------
    substrate: :py:class:`file`, :py:class:`io.BytesIO`
        BER/CER/DER serialization in form of a byte stream

    Keyword Args
    ------------
    asn1Spec: :py:class:`~pyasn1.type.base.PyAsn1Item`
        A pyasn1 type object to act as a template guiding the decoder.
        Depending on the ASN.1 structure being decoded, `asn1Spec` may
        or may not be required. One of the reasons why `asn1Spec` may
        me required is that ASN.1 structure is encoded in the *IMPLICIT*
        tagging mode.

    Yields
    ------
    : :py:class:`~pyasn1.type.base.PyAsn1Item`, :py:class:`~pyasn1.error.SubstrateUnderrunError`
        Decoded ASN.1 object (possibly, nested) or
        :py:class:`~pyasn1.error.SubstrateUnderrunError` object indicating
        insufficient BER/CER/DER serialization on input to fully recover ASN.1
        objects from it.
        
        In the latter case the caller is advised to ensure some more data in
        the input stream, then call the iterator again. The decoder will resume
        the decoding process using the newly arrived data.

        The `context` property of :py:class:`~pyasn1.error.SubstrateUnderrunError`
        object might hold a reference to the partially populated ASN.1 object
        being reconstructed.

    Raises
    ------
    ~pyasn1.error.PyAsn1Error, ~pyasn1.error.EndOfStreamError
        `PyAsn1Error` on deserialization error, `EndOfStreamError` on
         premature stream closure.

    Examples
    --------
    Decode BER serialisation without ASN.1 schema

    .. code-block:: pycon

        >>> stream = io.BytesIO(
        ...    b'0	\x02\x01\x01\x02\x01\x02\x02\x01\x03')
        >>>
        >>> for asn1Object in StreamingDecoder(stream):
        ...     print(asn1Object)
        >>>
        SequenceOf:
         1 2 3

    Decode BER serialisation with ASN.1 schema

    .. code-block:: pycon

        >>> stream = io.BytesIO(
        ...    b'0	\x02\x01\x01\x02\x01\x02\x02\x01\x03')
        >>>
        >>> schema = SequenceOf(componentType=Integer())
        >>>
        >>> decoder = StreamingDecoder(stream, asn1Spec=schema)
        >>> for asn1Object in decoder:
        ...     print(asn1Object)
        >>>
        SequenceOf:
         1 2 3
    """
    SINGLE_ITEM_DECODER = SingleItemDecoder

    def __init__(self, substrate, asn1Spec=None, **options):
        self._singleItemDecoder = self.SINGLE_ITEM_DECODER(**options)
        self._substrate = asSeekableStream(substrate)
        self._asn1Spec = asn1Spec
        self._options = options

    def __iter__(self):
        while True:
            for asn1Object in self._singleItemDecoder(self._substrate, self._asn1Spec, **self._options):
                yield asn1Object
            for chunk in isEndOfStream(self._substrate):
                if isinstance(chunk, SubstrateUnderrunError):
                    yield
                break
            if chunk:
                break