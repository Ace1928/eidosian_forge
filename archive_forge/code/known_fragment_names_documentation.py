from typing import Any
from ...error import GraphQLError
from ...language import FragmentSpreadNode
from . import ValidationRule
Known fragment names

    A GraphQL document is only valid if all ``...Fragment`` fragment spreads refer to
    fragments defined in the same document.

    See https://spec.graphql.org/draft/#sec-Fragment-spread-target-defined
    