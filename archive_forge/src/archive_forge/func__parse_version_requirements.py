from __future__ import absolute_import, division, print_function
import re
import importlib
import importlib.metadata
def _parse_version_requirements(val):
    """
    Parse package requirements.

    :param val: a string in the form ">=1.0.11,<2.0"
    :returns: list of tuples in the form [(">=", (1, 0, 11)), ("<", (2, 0, None))] or None if not parsed
    """
    reqs = []
    try:
        parts = val.split(',')
        for part in parts:
            match = re.match('\\s*(>=|<=|==|=|<|>|!=)\\s*([^\\s]+)', part)
            op = match.group(1)
            ver = match.group(2)
            ver_tuple = _parse_version(ver)
            if not ver_tuple:
                raise ValueError('invalid version {0}'.format(ver))
            reqs.append((op, ver_tuple))
        return reqs
    except Exception as e:
        raise ValueError("invalid version requirement '{0}' {1}".format(val, e))