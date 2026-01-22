import re
import sys
import six
from six.moves.urllib import parse as urlparse
from routes.util import _url_quote as url_quote, _str_encode, as_unicode
def _pathkeys(self, routepath):
    """Utility function to walk the route, and pull out the valid
        dynamic/wildcard keys."""
    collecting = False
    escaping = False
    current = ''
    done_on = ''
    var_type = ''
    just_started = False
    routelist = []
    for char in routepath:
        if escaping:
            if char in ['\\', ':', '*', '{', '}']:
                current += char
            else:
                current += '\\' + char
            escaping = False
        elif char == '\\':
            escaping = True
        elif char in [':', '*', '{'] and (not collecting) and (not self.static) or (char in ['{'] and (not collecting)):
            just_started = True
            collecting = True
            var_type = char
            if char == '{':
                done_on = '}'
                just_started = False
            if len(current) > 0:
                routelist.append(current)
                current = ''
        elif collecting and just_started:
            just_started = False
            if char == '(':
                done_on = ')'
            else:
                current = char
                done_on = self.done_chars + ('-',)
        elif collecting and char not in done_on:
            current += char
        elif collecting:
            collecting = False
            if var_type == '{':
                if current[0] == '.':
                    var_type = '.'
                    current = current[1:]
                else:
                    var_type = ':'
                opts = current.split(':')
                if len(opts) > 1:
                    current = opts[0]
                    self.reqs[current] = opts[1]
            routelist.append(dict(type=var_type, name=current))
            if char in self.done_chars:
                routelist.append(char)
            done_on = var_type = current = ''
        else:
            current += char
    if collecting:
        routelist.append(dict(type=var_type, name=current))
    elif current:
        routelist.append(current)
    return routelist