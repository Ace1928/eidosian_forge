from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
def add_virtual_authenticator(options: VirtualAuthenticatorOptions) -> typing.Generator[T_JSON_DICT, T_JSON_DICT, AuthenticatorId]:
    """
    Creates and adds a virtual authenticator.

    :param options:
    :returns: 
    """
    params: T_JSON_DICT = dict()
    params['options'] = options.to_json()
    cmd_dict: T_JSON_DICT = {'method': 'WebAuthn.addVirtualAuthenticator', 'params': params}
    json = (yield cmd_dict)
    return AuthenticatorId.from_json(json['authenticatorId'])