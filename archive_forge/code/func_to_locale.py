from contextlib import ContextDecorator
from decimal import ROUND_UP, Decimal
from django.utils.autoreload import autoreload_started, file_changed
from django.utils.functional import lazy
from django.utils.regex_helper import _lazy_re_compile
def to_locale(language):
    """Turn a language name (en-us) into a locale name (en_US)."""
    lang, _, country = language.lower().partition('-')
    if not country:
        return language[:3].lower() + language[3:]
    country, _, tail = country.partition('-')
    country = country.title() if len(country) > 2 else country.upper()
    if tail:
        country += '-' + tail
    return lang + '_' + country