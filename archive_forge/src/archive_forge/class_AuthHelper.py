from __future__ import annotations
import abc
import base64
import json
import os
import tempfile
import typing as t
from ..encoding import (
from ..io import (
from ..config import (
from ..util import (
class AuthHelper(metaclass=abc.ABCMeta):
    """Public key based authentication helper for Ansible Core CI."""

    def sign_request(self, request: dict[str, t.Any]) -> None:
        """Sign the given auth request and make the public key available."""
        payload_bytes = to_bytes(json.dumps(request, sort_keys=True))
        signature_raw_bytes = self.sign_bytes(payload_bytes)
        signature = to_text(base64.b64encode(signature_raw_bytes))
        request.update(signature=signature)

    def initialize_private_key(self) -> str:
        """
        Initialize and publish a new key pair (if needed) and return the private key.
        The private key is cached across ansible-test invocations, so it is only generated and published once per CI job.
        """
        path = os.path.expanduser('~/.ansible-core-ci-private.key')
        if os.path.exists(to_bytes(path)):
            private_key_pem = read_text_file(path)
        else:
            private_key_pem = self.generate_private_key()
            write_text_file(path, private_key_pem)
        return private_key_pem

    @abc.abstractmethod
    def sign_bytes(self, payload_bytes: bytes) -> bytes:
        """Sign the given payload and return the signature, initializing a new key pair if required."""

    @abc.abstractmethod
    def publish_public_key(self, public_key_pem: str) -> None:
        """Publish the given public key."""

    @abc.abstractmethod
    def generate_private_key(self) -> str:
        """Generate a new key pair, publishing the public key and returning the private key."""