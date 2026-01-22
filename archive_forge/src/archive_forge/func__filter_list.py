import contextlib
import fnmatch
import inspect
import re
import uuid
from decorator import decorator
import jmespath
import netifaces
from openstack import _log
from openstack import exceptions
def _filter_list(data, name_or_id, filters):
    """Filter a list by name/ID and arbitrary meta data.

    :param list data: The list of dictionary data to filter. It is expected
        that each dictionary contains an 'id' and 'name' key if a value for
        name_or_id is given.
    :param string name_or_id: The name or ID of the entity being filtered. Can
        be a glob pattern, such as 'nb01*'.
    :param filters: A dictionary of meta data to use for further filtering.
        Elements of this dictionary may, themselves, be dictionaries. Example::

            {
                'last_name': 'Smith',
                'other': {
                    'gender': 'Female'
                }
            }

        OR

        A string containing a jmespath expression for further filtering.
        Invalid filters will be ignored.
    """
    log = _log.setup_logging('openstack.fnmatch')
    if name_or_id:
        name_or_id = str(name_or_id)
        identifier_matches = []
        bad_pattern = False
        try:
            fn_reg = re.compile(fnmatch.translate(name_or_id))
        except re.error:
            fn_reg = None
        for e in data:
            e_id = str(e.get('id', None))
            e_name = str(e.get('name', None))
            if e_id and e_id == name_or_id or (e_name and e_name == name_or_id):
                identifier_matches.append(e)
            else:
                if not fn_reg:
                    bad_pattern = True
                    continue
                if e_id and fn_reg.match(e_id) or (e_name and fn_reg.match(e_name)):
                    identifier_matches.append(e)
        if not identifier_matches and bad_pattern:
            log.debug('Bad pattern passed to fnmatch', exc_info=True)
        data = identifier_matches
    if not filters:
        return data
    if isinstance(filters, str):
        return jmespath.search(filters, data)

    def _dict_filter(f, d):
        if not d:
            return False
        for key in f.keys():
            if key not in d:
                log.warning('Invalid filter: %s is not an attribute of %s.%s', key, e.__class__.__module__, e.__class__.__qualname__)
                raise AttributeError(key)
            if isinstance(f[key], dict):
                if not _dict_filter(f[key], d.get(key, None)):
                    return False
            elif d.get(key, None) != f[key]:
                return False
        return True
    filtered = []
    for e in data:
        if _dict_filter(filters, e):
            filtered.append(e)
    return filtered