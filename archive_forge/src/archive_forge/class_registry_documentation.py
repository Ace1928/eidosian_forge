from __future__ import annotations
import typing
from marshmallow.exceptions import RegistryError
Retrieve a class from the registry.

    :raises: marshmallow.exceptions.RegistryError if the class cannot be found
        or if there are multiple entries for the given class name.
    