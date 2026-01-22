import textwrap
import warnings
class UnofficialUsageWarning(Warning):
    """Use of unofficial service-types is discouraged."""
    details = '\n    Requested service_type {given} is not a known official OpenStack project.\n    '