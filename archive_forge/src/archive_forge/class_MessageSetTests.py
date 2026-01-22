from __future__ import annotations
import base64
import codecs
import functools
import locale
import os
import uuid
from collections import OrderedDict
from io import BytesIO
from itertools import chain
from typing import Optional, Type
from unittest import skipIf
from zope.interface import implementer
from zope.interface.verify import verifyClass, verifyObject
from twisted.cred.checkers import InMemoryUsernamePasswordDatabaseDontUse
from twisted.cred.credentials import (
from twisted.cred.error import UnauthorizedLogin
from twisted.cred.portal import IRealm, Portal
from twisted.internet import defer, error, interfaces, reactor
from twisted.internet.defer import Deferred
from twisted.internet.task import Clock
from twisted.internet.testing import StringTransport, StringTransportWithDisconnection
from twisted.mail import imap4
from twisted.mail.imap4 import MessageSet
from twisted.mail.interfaces import (
from twisted.protocols import loopback
from twisted.python import failure, log, util
from twisted.python.compat import iterbytes, nativeString, networkString
from twisted.trial.unittest import SynchronousTestCase, TestCase
class MessageSetTests(SynchronousTestCase):
    """
    Tests for L{MessageSet}.
    """

    def test_equalityIterationAndAddition(self):
        """
        Test the following properties of L{MessageSet} addition and
        equality:

            1. Two empty L{MessageSet}s are equal to each other;

            2. A L{MessageSet} is not equal to any other object;

            2. Adding a L{MessageSet} and another L{MessageSet} or an
               L{int} representing a single message or a sequence of
               L{int}s representing a sequence of message numbers
               produces a new L{MessageSet} that:

            3. Has a length equal to the number of messages within
               each sequence of message numbers;

            4. Yields each message number in ascending order when
               iterated over;

            6. L{MessageSet.add} with a single message or a start and
               end message satisfies 3 and 4 above.
        """
        m1 = MessageSet()
        m2 = MessageSet()
        self.assertEqual(m1, m2)
        self.assertNotEqual(m1, ())
        m1 = m1 + 1
        self.assertEqual(len(m1), 1)
        self.assertEqual(list(m1), [1])
        m1 = m1 + (1, 3)
        self.assertEqual(len(m1), 3)
        self.assertEqual(list(m1), [1, 2, 3])
        m2 = m2 + (1, 3)
        self.assertEqual(m1, m2)
        self.assertEqual(list(m1 + m2), [1, 2, 3])
        m1.add(5)
        self.assertEqual(len(m1), 4)
        self.assertEqual(list(m1), [1, 2, 3, 5])
        self.assertNotEqual(m1, m2)
        m1.add(6, 8)
        self.assertEqual(len(m1), 7)
        self.assertEqual(list(m1), [1, 2, 3, 5, 6, 7, 8])

    def test_lengthWithWildcardRange(self):
        """
        A L{MessageSet} that has a range that ends with L{None} raises
        a L{TypeError} when its length is requested.
        """
        self.assertRaises(TypeError, len, MessageSet(1, None))

    def test_reprSanity(self):
        """
        L{MessageSet.__repr__} does not raise an exception
        """
        repr(MessageSet(1, 2))

    def test_stringRepresentationWithWildcards(self):
        """
        In a L{MessageSet}, in the presence of wildcards, if the
        highest message id is known, the wildcard should get replaced
        by that high value.
        """
        inputs = [imap4.parseIdList(b'*'), imap4.parseIdList(b'1:*'), imap4.parseIdList(b'3:*', 6), imap4.parseIdList(b'*:2', 6)]
        outputs = ['*', '1:*', '3:6', '2:6']
        for i, o in zip(inputs, outputs):
            self.assertEqual(str(i), o)

    def test_stringRepresentationWithInversion(self):
        """
        In a L{MessageSet}, inverting the high and low numbers in a
        range doesn't affect the meaning of the range.  For example,
        3:2 displays just like 2:3, because according to the RFC they
        have the same meaning.
        """
        inputs = [imap4.parseIdList(b'2:3'), imap4.parseIdList(b'3:2')]
        outputs = ['2:3', '2:3']
        for i, o in zip(inputs, outputs):
            self.assertEqual(str(i), o)

    def test_createWithSingleMessageNumber(self):
        """
        Creating a L{MessageSet} with a single message number adds
        only that message to the L{MessageSet}; its serialized form
        includes only that message number, its length is one, and it
        yields only that message number.
        """
        m = MessageSet(1)
        self.assertEqual(str(m), '1')
        self.assertEqual(len(m), 1)
        self.assertEqual(list(m), [1])

    def test_createWithSequence(self):
        """
        Creating a L{MessageSet} with both a start and end message
        number adds the sequence between to the L{MessageSet}; its
        serialized form consists that range, its length is the length
        of the sequence, and it yields the message numbers inclusively
        between the start and end.
        """
        m = MessageSet(1, 10)
        self.assertEqual(str(m), '1:10')
        self.assertEqual(len(m), 10)
        self.assertEqual(list(m), [1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

    def test_createWithSingleWildcard(self):
        """
        Creating a L{MessageSet} with a single L{None}, representing
        C{*}, adds C{*} to the range; its serialized form includes
        only C{*}, its length is one, but it cannot be iterated over
        because its endpoint is unknown.
        """
        m = MessageSet(None)
        self.assertEqual(str(m), '*')
        self.assertEqual(len(m), 1)
        self.assertRaises(TypeError, list, m)

    def test_setLastSingleWildcard(self):
        """
        Setting L{MessageSet.last} replaces L{None}, representing
        C{*}, with that number, making that L{MessageSet} iterable.
        """
        singleMessageReplaced = MessageSet(None)
        singleMessageReplaced.last = 10
        self.assertEqual(list(singleMessageReplaced), [10])
        rangeReplaced = MessageSet(3, None)
        rangeReplaced.last = 1
        self.assertEqual(list(rangeReplaced), [1, 2, 3])

    def test_setLastWithWildcardRange(self):
        """
        Setting L{MessageSet.last} replaces L{None} in all ranges.
        """
        m = MessageSet(1, None)
        m.add(2, None)
        m.last = 5
        self.assertEqual(list(m), [1, 2, 3, 4, 5])

    def test_setLastTwiceFails(self):
        """
        L{MessageSet.last} cannot be set twice.
        """
        m = MessageSet(1, None)
        m.last = 2
        with self.assertRaises(ValueError):
            m.last = 3

    def test_lastOverridesNoneInAdd(self):
        """
        Adding a L{None}, representing C{*}, or a sequence that
        includes L{None} to a L{MessageSet} whose
        L{last<MessageSet.last>} property has been set replaces all
        occurrences of L{None} with the value of
        L{last<MessageSet.last>}.
        """
        hasLast = MessageSet(1)
        hasLast.last = 4
        hasLast.add(None)
        self.assertEqual(list(hasLast), [1, 4])
        self.assertEqual(list(hasLast + (None, 5)), [1, 4, 5])
        hasLast.add(3, None)
        self.assertEqual(list(hasLast), [1, 3, 4])

    def test_getLast(self):
        """
        Accessing L{MessageSet.last} returns the last value.
        """
        m = MessageSet(1, None)
        m.last = 2
        self.assertEqual(m.last, 2)

    def test_extend(self):
        """
        L{MessageSet.extend} accepts as its arugment an L{int} or
        L{None}, or a sequence L{int}s or L{None}s of length two, or
        another L{MessageSet}, combining its argument with its
        instance's existing ranges.
        """
        extendWithInt = MessageSet()
        extendWithInt.extend(1)
        self.assertEqual(list(extendWithInt), [1])
        extendWithNone = MessageSet()
        extendWithNone.extend(None)
        self.assertEqual(str(extendWithNone), '*')
        extendWithSequenceOfInts = MessageSet()
        extendWithSequenceOfInts.extend((1, 3))
        self.assertEqual(list(extendWithSequenceOfInts), [1, 2, 3])
        extendWithSequenceOfNones = MessageSet()
        extendWithSequenceOfNones.extend((None, None))
        self.assertEqual(str(extendWithSequenceOfNones), '*')
        extendWithMessageSet = MessageSet()
        extendWithMessageSet.extend(MessageSet(1, 3))
        self.assertEqual(list(extendWithMessageSet), [1, 2, 3])

    def test_contains(self):
        """
        A L{MessageSet} contains a number if the number falls within
        one of its ranges, and raises L{TypeError} if any range
        contains L{None}.
        """
        hasFive = MessageSet(1, 7)
        doesNotHaveFive = MessageSet(1, 4) + MessageSet(6, 7)
        self.assertIn(5, hasFive)
        self.assertNotIn(5, doesNotHaveFive)
        hasFiveButHasNone = hasFive + None
        with self.assertRaises(TypeError):
            5 in hasFiveButHasNone
        hasFiveButHasNoneInSequence = hasFive + (10, 12)
        hasFiveButHasNoneInSequence.add(8, None)
        with self.assertRaises(TypeError):
            5 in hasFiveButHasNoneInSequence

    def test_rangesMerged(self):
        """
        Adding a sequence of message numbers to a L{MessageSet} that
        begins or ends immediately before or after an existing
        sequence in that L{MessageSet}, or overlaps one, merges the two.
        """
        mergeAfter = MessageSet(1, 3)
        mergeBefore = MessageSet(6, 8)
        mergeBetweenSequence = mergeAfter + mergeBefore
        mergeBetweenNumber = mergeAfter + MessageSet(5, 7)
        self.assertEqual(list(mergeAfter + (2, 4)), [1, 2, 3, 4])
        self.assertEqual(list(mergeAfter + (3, 5)), [1, 2, 3, 4, 5])
        self.assertEqual(list(mergeBefore + (5, 7)), [5, 6, 7, 8])
        self.assertEqual(list(mergeBefore + (4, 6)), [4, 5, 6, 7, 8])
        self.assertEqual(list(mergeBetweenSequence + (3, 5)), [1, 2, 3, 4, 5, 6, 7, 8])
        self.assertEqual(list(mergeBetweenNumber + MessageSet(4)), [1, 2, 3, 4, 5, 6, 7])

    def test_seq_rangeExamples(self):
        """
        Test the C{seq-range} examples from Section 9, "Formal Syntax"
        of RFC 3501::

            Example: 2:4 and 4:2 are equivalent and indicate values
                     2, 3, and 4.

            Example: a unique identifier sequence range of
                     3291:* includes the UID of the last message in
                     the mailbox, even if that value is less than 3291.

        @see: U{http://tools.ietf.org/html/rfc3501#section-9}
        """
        self.assertEqual(MessageSet(2, 4), MessageSet(4, 2))
        self.assertEqual(list(MessageSet(2, 4)), [2, 3, 4])
        m = MessageSet(3291, None)
        m.last = 3290
        self.assertEqual(list(m), [3290, 3291])

    def test_sequence_setExamples(self):
        """
        Test the C{sequence-set} examples from Section 9, "Formal
        Syntax" of RFC 3501.  In particular, L{MessageSet} reorders
        and coalesces overlaps::

            Example: a message sequence number set of
                     2,4:7,9,12:* for a mailbox with 15 messages is
                     equivalent to 2,4,5,6,7,9,12,13,14,15

            Example: a message sequence number set of *:4,5:7
                     for a mailbox with 10 messages is equivalent to
                     10,9,8,7,6,5,4,5,6,7 and MAY be reordered and
                     overlap coalesced to be 4,5,6,7,8,9,10.

        @see: U{http://tools.ietf.org/html/rfc3501#section-9}
        """
        fromFifteenMessages = MessageSet(2) + MessageSet(4, 7) + MessageSet(9) + MessageSet(12, None)
        fromFifteenMessages.last = 15
        self.assertEqual(','.join((str(i) for i in fromFifteenMessages)), '2,4,5,6,7,9,12,13,14,15')
        fromTenMessages = MessageSet(None, 4) + MessageSet(5, 7)
        fromTenMessages.last = 10
        self.assertEqual(','.join((str(i) for i in fromTenMessages)), '4,5,6,7,8,9,10')