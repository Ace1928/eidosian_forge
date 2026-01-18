from neutron_lib.api import converters
from neutron_lib.api.definitions import network
from neutron_lib.api import validators
from neutron_lib.api.validators import multiprovidernet as mp_validator
from neutron_lib import constants
from neutron_lib.exceptions import multiprovidernet as mp_exc
Helper function checking duplicate segments.

    If is_partial_funcs is specified and not None, then
    SegmentsContainDuplicateEntry is raised if two segments are identical and
    non partially defined (is_partial_func(segment) == False).
    Otherwise SegmentsContainDuplicateEntry is raised if two segment are
    identical.
    