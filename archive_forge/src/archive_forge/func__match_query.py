import random
import re
import shelve
import ldap
from oslo_log import log
import keystone.conf
from keystone import exception
from keystone.identity.backends.ldap import common
def _match_query(query, attrs, attrs_checked):
    """Match an ldap query to an attribute dictionary.

    The characters &, |, and ! are supported in the query. No syntax checking
    is performed, so malformed queries will not work correctly.
    """
    inner = query[1:-1]
    if inner.startswith(('&', '|')):
        if inner[0] == '&':
            matchfn = all
        else:
            matchfn = any
        groups = _paren_groups(inner[1:])
        return matchfn((_match_query(group, attrs, attrs_checked) for group in groups))
    if inner.startswith('!'):
        return not _match_query(query[2:-1], attrs, attrs_checked)
    k, _sep, v = inner.partition('=')
    attrs_checked.add(k.lower())
    return _match(k, v, attrs)