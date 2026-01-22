import errno
import smtplib
import socket
from email.message import Message
from breezy import config, email_message, smtp_connection, tests, ui
class StubSMTPFactory:
    """A fake SMTP connection to test the connection setup."""

    def __init__(self, fail_on=None, smtp_features=None):
        self._fail_on = fail_on or []
        self._calls = []
        self._smtp_features = smtp_features or []
        self._ehlo_called = False

    def __call__(self, host='localhost'):
        self._calls.append(('connect', host))
        return self

    def connect(self, server):
        raise NotImplementedError

    def helo(self):
        self._calls.append(('helo',))
        if 'helo' in self._fail_on:
            return (500, 'helo failure')
        else:
            return (200, 'helo success')

    def ehlo(self):
        self._calls.append(('ehlo',))
        if 'ehlo' in self._fail_on:
            return (500, 'ehlo failure')
        else:
            self._ehlo_called = True
            return (200, 'ehlo success')

    def has_extn(self, extension):
        self._calls.append(('has_extn', extension))
        return self._ehlo_called and extension in self._smtp_features

    def starttls(self):
        self._calls.append(('starttls',))
        if 'starttls' in self._fail_on:
            return (500, 'starttls failure')
        else:
            self._ehlo_called = True
            return (200, 'starttls success')