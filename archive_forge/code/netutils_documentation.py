import netaddr
import re
from heat.common.i18n import _
Check whether orig_prefixes is subset of new_prefixes.


    This takes valid prefix lists for orig_prefixes and new_prefixes,
    returns 'True', if orig_prefixes is subset of new_prefixes.
    