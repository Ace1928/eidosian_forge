from __future__ import annotations
import email.message
import email.parser
from io import BytesIO, StringIO
from typing import IO, AnyStr, Callable
from twisted.mail import bounce
from twisted.trial import unittest

        L{twisted.mail.bounce.generateBounce} with big L{unicode} and
        L{bytes} messages.
        