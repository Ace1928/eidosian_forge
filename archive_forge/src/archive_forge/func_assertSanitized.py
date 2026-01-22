from __future__ import annotations
from typing import Sequence
from twisted.trial.unittest import TestCase
from twisted.web.http_headers import Headers
from twisted.web.test.requesthelper import (
def assertSanitized(testCase: TestCase, components: Sequence[bytes] | Sequence[str], expected: bytes) -> None:
    """
    Assert that the components are sanitized to the expected value as
    both a header name and value, across all of L{Header}'s setters
    and getters.

    @param testCase: A test case.

    @param components: A sequence of values that contain linear
        whitespace to use as header names and values; see
        C{textLinearWhitespaceComponents} and
        C{bytesLinearWhitespaceComponents}

    @param expected: The expected sanitized form of the component for
        both headers names and their values.
    """
    for component in components:
        headers = []
        headers.append(Headers({component: [component]}))
        added = Headers()
        added.addRawHeader(component, component)
        headers.append(added)
        setHeader = Headers()
        setHeader.setRawHeaders(component, [component])
        headers.append(setHeader)
        for header in headers:
            testCase.assertEqual(list(header.getAllRawHeaders()), [(expected, [expected])])
            testCase.assertEqual(header.getRawHeaders(expected), [expected])