from operator import itemgetter
from typing import Any, List, Tuple, Dict, Sequence, Iterable, Optional, Mapping, Union
import warnings
import sys
from langcodes.tag_parser import LanguageTagError, parse_tag, normalize_characters
from langcodes.language_distance import tuple_distance_cached
from langcodes.data_dicts import (
def to_alpha3(self, variant: str='T') -> str:
    """
        Get the three-letter language code for this language, even if it's
        canonically written with a two-letter code.

        These codes are the 'alpha3' codes defined by ISO 639-2.

        When this function returns, it always returns a 3-letter string. If
        there is no known alpha3 code for the language, it raises a LookupError.

        In cases where the distinction matters, we default to the 'terminology'
        code. You can pass `variant='B'` to get the 'bibliographic' code instead.
        For example, the terminology code for German is 'deu', while the
        bibliographic code is 'ger'.

        (The confusion between these two sets of codes is a good reason to avoid
        using alpha3 codes. Every language that has two different alpha3 codes
        also has an alpha2 code that's preferred, such as 'de' for German.)

        >>> Language.get('fr').to_alpha3()
        'fra'
        >>> Language.get('fr-CA').to_alpha3()
        'fra'
        >>> Language.get('fr').to_alpha3(variant='B')
        'fre'
        >>> Language.get('de').to_alpha3(variant='T')
        'deu'
        >>> Language.get('ja').to_alpha3()
        'jpn'
        >>> Language.get('un').to_alpha3()
        Traceback (most recent call last):
            ...
        LookupError: 'un' is not a known language code, and has no alpha3 code.


        All valid two-letter language codes have corresponding alpha3 codes,
        even the un-normalized ones. If they were assigned an alpha3 code by ISO
        before they were assigned a normalized code by CLDR, these codes may be
        different:

        >>> Language.get('tl', normalize=False).to_alpha3()
        'tgl'
        >>> Language.get('tl').to_alpha3()
        'fil'
        >>> Language.get('sh', normalize=False).to_alpha3()
        'hbs'


        Three-letter codes are preserved, even if they're unknown:

        >>> Language.get('qqq').to_alpha3()
        'qqq'
        >>> Language.get('und').to_alpha3()
        'und'
        """
    variant = variant.upper()
    if variant not in 'BT':
        raise ValueError("Variant must be 'B' or 'T'")
    language = self.language
    if language is None:
        return 'und'
    elif len(language) == 3:
        return language
    elif variant == 'B' and language in LANGUAGE_ALPHA3_BIBLIOGRAPHIC:
        return LANGUAGE_ALPHA3_BIBLIOGRAPHIC[language]
    elif language in LANGUAGE_ALPHA3:
        return LANGUAGE_ALPHA3[language]
    else:
        raise LookupError(f'{language!r} is not a known language code, and has no alpha3 code.')