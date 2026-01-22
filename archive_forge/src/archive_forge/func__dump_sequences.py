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
def _dump_sequences(self, message, key):
    """Inspect a new MHMessage and update sequences appropriately."""
    pending_sequences = message.get_sequences()
    all_sequences = self.get_sequences()
    for name, key_list in all_sequences.items():
        if name in pending_sequences:
            key_list.append(key)
        elif key in key_list:
            del key_list[key_list.index(key)]
    for sequence in pending_sequences:
        if sequence not in all_sequences:
            all_sequences[sequence] = [key]
    self.set_sequences(all_sequences)