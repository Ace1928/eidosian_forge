import copy
from googlecloudsdk.calliope.concepts import deps as deps_lib
def _PluralizeFallthrough(fallthrough):
    plural_fallthrough = copy.deepcopy(fallthrough)
    plural_fallthrough.plural = True
    return plural_fallthrough