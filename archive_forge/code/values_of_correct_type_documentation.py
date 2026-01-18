from typing import cast, Any
from ...error import GraphQLError
from ...language import (
from ...pyutils import did_you_mean, suggestion_list, Undefined
from ...type import (
from . import ValidationRule
Check whether this is a valid value node.

        Any value literal may be a valid representation of a Scalar, depending on that
        scalar type.
        