from sys import exc_info
from typing import Any, Collection, Dict, List, Optional, Union, TYPE_CHECKING
Get error formatted according to the specification.

        Given a GraphQLError, format it according to the rules described by the
        "Response Format, Errors" section of the GraphQL Specification.
        