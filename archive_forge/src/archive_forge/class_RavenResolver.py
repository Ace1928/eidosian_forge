from __future__ import absolute_import
import re
from sentry_sdk._types import TYPE_CHECKING
from django import VERSION as DJANGO_VERSION
class RavenResolver(object):
    _new_style_group_matcher = re.compile('<(?:([^>:]+):)?([^>]+)>')
    _optional_group_matcher = re.compile('\\(\\?\\:([^\\)]+)\\)')
    _named_group_matcher = re.compile('\\(\\?P<(\\w+)>[^\\)]+\\)+')
    _non_named_group_matcher = re.compile('\\([^\\)]+\\)')
    _either_option_matcher = re.compile('\\[([^\\]]+)\\|([^\\]]+)\\]')
    _camel_re = re.compile('([A-Z]+)([a-z])')
    _cache = {}

    def _simplify(self, pattern):
        """
        Clean up urlpattern regexes into something readable by humans:

        From:
        > "^(?P<sport_slug>\\w+)/athletes/(?P<athlete_slug>\\w+)/$"

        To:
        > "{sport_slug}/athletes/{athlete_slug}/"
        """
        if RoutePattern is not None and hasattr(pattern, 'pattern') and isinstance(pattern.pattern, RoutePattern):
            return self._new_style_group_matcher.sub(lambda m: '{%s}' % m.group(2), pattern.pattern._route)
        result = get_regex(pattern).pattern
        result = self._optional_group_matcher.sub(lambda m: '%s' % m.group(1), result)
        result = self._named_group_matcher.sub(lambda m: '{%s}' % m.group(1), result)
        result = self._non_named_group_matcher.sub('{var}', result)
        result = self._either_option_matcher.sub(lambda m: m.group(1), result)
        result = result.replace('^', '').replace('$', '').replace('?', '').replace('\\A', '').replace('\\Z', '').replace('//', '/').replace('\\', '')
        return result

    def _resolve(self, resolver, path, parents=None):
        match = get_regex(resolver).search(path)
        if not match:
            return None
        if parents is None:
            parents = [resolver]
        elif resolver not in parents:
            parents = parents + [resolver]
        new_path = path[match.end():]
        for pattern in resolver.url_patterns:
            if not pattern.callback:
                match_ = self._resolve(pattern, new_path, parents)
                if match_:
                    return match_
                continue
            elif not get_regex(pattern).search(new_path):
                continue
            try:
                return self._cache[pattern]
            except KeyError:
                pass
            prefix = ''.join((self._simplify(p) for p in parents))
            result = prefix + self._simplify(pattern)
            if not result.startswith('/'):
                result = '/' + result
            self._cache[pattern] = result
            return result
        return None

    def resolve(self, path, urlconf=None):
        resolver = get_resolver(urlconf)
        match = self._resolve(resolver, path)
        return match