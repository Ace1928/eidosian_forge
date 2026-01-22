from __future__ import annotations
import base64
import collections.abc
import logging
import os
import typing as t
from spnego._context import (
from spnego._credential import Credential, CredentialCache, Password, unify_credentials
from spnego.channel_bindings import GssChannelBindings
from spnego.exceptions import (
from spnego.exceptions import WinError as NativeError
from spnego.iov import BufferType, IOVBuffer, IOVResBuffer
def _get_native_bindings(self, channel_bindings: GssChannelBindings) -> sspilib.SecChannelBindings:
    """Gets the raw byte value of the SEC_CHANNEL_BINDINGS structure."""
    return sspilib.SecChannelBindings(initiator_addr_type=int(channel_bindings.initiator_addrtype), initiator_addr=channel_bindings.initiator_address, acceptor_addr_type=int(channel_bindings.acceptor_addrtype), acceptor_addr=channel_bindings.acceptor_address, application_data=channel_bindings.application_data)