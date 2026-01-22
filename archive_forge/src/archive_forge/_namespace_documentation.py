import collections
from ._caveat import error_caveat
from ._utils import condition_with_prefix
 Resolves the given caveat(string) by using resolve to map from its
        schema namespace to the appropriate prefix.
        If there is no registered prefix for the namespace, it returns an error
        caveat.
        If cav.namespace is empty or cav.location is non-empty, it returns cav
        unchanged.

        It does not mutate ns and may be called concurrently with other
        non-mutating Namespace methods.
        :return: Caveat object
        