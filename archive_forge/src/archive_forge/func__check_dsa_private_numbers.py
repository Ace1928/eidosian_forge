from __future__ import annotations
import abc
import typing
from cryptography.hazmat.bindings._rust import openssl as rust_openssl
from cryptography.hazmat.primitives import _serialization, hashes
from cryptography.hazmat.primitives.asymmetric import utils as asym_utils
def _check_dsa_private_numbers(numbers: DSAPrivateNumbers) -> None:
    parameters = numbers.public_numbers.parameter_numbers
    _check_dsa_parameters(parameters)
    if numbers.x <= 0 or numbers.x >= parameters.q:
        raise ValueError('x must be > 0 and < q.')
    if numbers.public_numbers.y != pow(parameters.g, numbers.x, parameters.p):
        raise ValueError('y must be equal to (g ** x % p).')