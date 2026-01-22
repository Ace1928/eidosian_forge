import binascii
import codecs
import copy
import email.utils
import functools
import re
import string
import tempfile
import time
import uuid
from base64 import decodebytes, encodebytes
from io import BytesIO
from itertools import chain
from typing import Any, List, cast
from zope.interface import implementer
from twisted.cred import credentials
from twisted.cred.error import UnauthorizedLogin, UnhandledCredentials
from twisted.internet import defer, error, interfaces
from twisted.internet.defer import maybeDeferred
from twisted.mail._cred import (
from twisted.mail._except import (
from twisted.mail.interfaces import (
from twisted.protocols import basic, policies
from twisted.python import log, text
from twisted.python.compat import (
def _parseFetchPairs(self, fetchResponseList):
    """
        Given the result of parsing a single I{FETCH} response, construct a
        L{dict} mapping response keys to response values.

        @param fetchResponseList: The result of parsing a I{FETCH} response
            with L{parseNestedParens} and extracting just the response data
            (that is, just the part that comes after C{"FETCH"}).  The form
            of this input (and therefore the output of this method) is very
            disagreeable.  A valuable improvement would be to enumerate the
            possible keys (representing them as structured objects of some
            sort) rather than using strings and tuples of tuples of strings
            and so forth.  This would allow the keys to be documented more
            easily and would allow for a much simpler application-facing API
            (one not based on looking up somewhat hard to predict keys in a
            dict).  Since C{fetchResponseList} notionally represents a
            flattened sequence of pairs (identifying keys followed by their
            associated values), collapsing such complex elements of this
            list as C{["BODY", ["HEADER.FIELDS", ["SUBJECT"]]]} into a
            single object would also greatly simplify the implementation of
            this method.

        @return: A C{dict} of the response data represented by C{pairs}.  Keys
            in this dictionary are things like C{"RFC822.TEXT"}, C{"FLAGS"}, or
            C{("BODY", ("HEADER.FIELDS", ("SUBJECT",)))}.  Values are entirely
            dependent on the key with which they are associated, but retain the
            same structured as produced by L{parseNestedParens}.
        """

    def nativeStringResponse(thing):
        if isinstance(thing, bytes):
            return thing.decode('charmap')
        elif isinstance(thing, list):
            return [nativeStringResponse(subthing) for subthing in thing]
    values = {}
    unstructured = []
    responseParts = iter(fetchResponseList)
    while True:
        try:
            key = next(responseParts)
        except StopIteration:
            break
        try:
            value = next(responseParts)
        except StopIteration:
            raise IllegalServerResponse(b'Not enough arguments', fetchResponseList)
        if key not in (b'BODY', b'BODY.PEEK'):
            hasSection = False
        elif not isinstance(value, list):
            hasSection = False
        elif len(value) > 2:
            hasSection = False
        elif value and isinstance(value[0], list):
            hasSection = False
        else:
            hasSection = True
        key = nativeString(key)
        unstructured.append(key)
        if hasSection:
            if len(value) < 2:
                value = [nativeString(v) for v in value]
                unstructured.append(value)
                key = (key, tuple(value))
            else:
                valueHead = nativeString(value[0])
                valueTail = [nativeString(v) for v in value[1]]
                unstructured.append([valueHead, valueTail])
                key = (key, (valueHead, tuple(valueTail)))
            try:
                value = next(responseParts)
            except StopIteration:
                raise IllegalServerResponse(b'Not enough arguments', fetchResponseList)
            if value.startswith(b'<') and value.endswith(b'>'):
                try:
                    int(value[1:-1])
                except ValueError:
                    pass
                else:
                    value = nativeString(value)
                    unstructured.append(value)
                    key = key + (value,)
                    try:
                        value = next(responseParts)
                    except StopIteration:
                        raise IllegalServerResponse(b'Not enough arguments', fetchResponseList)
        value = nativeStringResponse(value)
        unstructured.append(value)
        values[key] = value
    return (values, unstructured)