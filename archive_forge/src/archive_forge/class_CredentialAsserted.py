from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
@event_class('WebAuthn.credentialAsserted')
@dataclass
class CredentialAsserted:
    """
    Triggered when a credential is used in a webauthn assertion.
    """
    authenticator_id: AuthenticatorId
    credential: Credential

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> CredentialAsserted:
        return cls(authenticator_id=AuthenticatorId.from_json(json['authenticatorId']), credential=Credential.from_json(json['credential']))