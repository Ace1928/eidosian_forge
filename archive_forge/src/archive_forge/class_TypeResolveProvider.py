import abc
from typing import Any
class TypeResolveProvider(_AbstractResolver, _AbstractProvider):
    """
    Implement this in an extension to provide a custom resolver, see _AbstractResolver
    """