from __future__ import division
import contextlib
import json
import numbers
from jsonschema import _utils, _validators
from jsonschema.compat import (
from jsonschema.exceptions import ErrorTree  # Backwards compat  # noqa: F401
from jsonschema.exceptions import RefResolutionError, SchemaError, UnknownType
def validates(version):
    """
    Register the decorated validator for a ``version`` of the specification.

    Registered validators and their meta schemas will be considered when
    parsing ``$schema`` properties' URIs.

    Arguments:

        version (str):

            An identifier to use as the version's name

    Returns:

        callable: a class decorator to decorate the validator with the version

    """

    def _validates(cls):
        validators[version] = cls
        if u'id' in cls.META_SCHEMA:
            meta_schemas[cls.META_SCHEMA[u'id']] = cls
        return cls
    return _validates