from __future__ import annotations
import codecs
import re
from .structures import ImmutableList
class LanguageAccept(Accept):
    """Like :class:`Accept` but with normalization for language tags."""

    def _value_matches(self, value, item):
        return item == '*' or _normalize_lang(value) == _normalize_lang(item)

    def best_match(self, matches, default=None):
        """Given a list of supported values, finds the best match from
        the list of accepted values.

        Language tags are normalized for the purpose of matching, but
        are returned unchanged.

        If no exact match is found, this will fall back to matching
        the first subtag (primary language only), first with the
        accepted values then with the match values. This partial is not
        applied to any other language subtags.

        The default is returned if no exact or fallback match is found.

        :param matches: A list of supported languages to find a match.
        :param default: The value that is returned if none match.
        """
        result = super().best_match(matches)
        if result is not None:
            return result
        fallback = Accept([(_locale_delim_re.split(item[0], 1)[0], item[1]) for item in self])
        result = fallback.best_match(matches)
        if result is not None:
            return result
        fallback_matches = [_locale_delim_re.split(item, 1)[0] for item in matches]
        result = super().best_match(fallback_matches)
        if result is not None:
            return next((item for item in matches if item.startswith(result)))
        return default