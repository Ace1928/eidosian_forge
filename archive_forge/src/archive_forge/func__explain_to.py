import os
import time
import calendar
import socket
import errno
import copy
import warnings
import email
import email.message
import email.generator
import io
import contextlib
from types import GenericAlias
def _explain_to(self, message):
    """Copy Babyl-specific state to message insofar as possible."""
    if isinstance(message, MaildirMessage):
        labels = set(self.get_labels())
        if 'unseen' in labels:
            message.set_subdir('cur')
        else:
            message.set_subdir('cur')
            message.add_flag('S')
        if 'forwarded' in labels or 'resent' in labels:
            message.add_flag('P')
        if 'answered' in labels:
            message.add_flag('R')
        if 'deleted' in labels:
            message.add_flag('T')
    elif isinstance(message, _mboxMMDFMessage):
        labels = set(self.get_labels())
        if 'unseen' not in labels:
            message.add_flag('RO')
        else:
            message.add_flag('O')
        if 'deleted' in labels:
            message.add_flag('D')
        if 'answered' in labels:
            message.add_flag('A')
    elif isinstance(message, MHMessage):
        labels = set(self.get_labels())
        if 'unseen' in labels:
            message.add_sequence('unseen')
        if 'answered' in labels:
            message.add_sequence('replied')
    elif isinstance(message, BabylMessage):
        message.set_visible(self.get_visible())
        for label in self.get_labels():
            message.add_label(label)
    elif isinstance(message, Message):
        pass
    else:
        raise TypeError('Cannot convert to specified type: %s' % type(message))