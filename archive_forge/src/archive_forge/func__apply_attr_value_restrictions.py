import copy
import importlib
import logging
import re
from warnings import warn as _warn
from saml2 import saml
from saml2 import xmlenc
from saml2.attribute_converter import ac_factory
from saml2.attribute_converter import from_local
from saml2.attribute_converter import get_local_name
from saml2.s_utils import MissingValue
from saml2.s_utils import assertion_factory
from saml2.s_utils import factory
from saml2.s_utils import sid
from saml2.saml import NAME_FORMAT_URI
from saml2.time_util import in_a_while
from saml2.time_util import instant
def _apply_attr_value_restrictions(attr, res, must=False):
    values = [av['text'] for av in attr.get('attribute_value', [])]
    try:
        res[_fn].extend(_filter_values(ava[_fn], values))
    except KeyError:
        val = _filter_values(ava[_fn], values)
        res[_fn] = val if val is not None else []
    return _filter_values(ava[_fn], values, must)