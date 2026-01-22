import sys
import os
import time
import re
import string
import urllib.request, urllib.parse, urllib.error
from docutils import frontend, nodes, languages, writers, utils, io
from docutils.utils.error_reporting import SafeString
from docutils.transforms import writer_aux
from docutils.utils.math import pick_math_environment, unichar2tex
class Babel(object):
    """Language specifics for LaTeX."""
    language_codes = {'af': 'afrikaans', 'ar': 'arabic', 'bg': 'bulgarian', 'br': 'breton', 'ca': 'catalan', 'cs': 'czech', 'cy': 'welsh', 'da': 'danish', 'de': 'ngerman', 'de-1901': 'german', 'de-AT': 'naustrian', 'de-AT-1901': 'austrian', 'dsb': 'lowersorbian', 'el': 'greek', 'el-polyton': 'polutonikogreek', 'en': 'english', 'en-AU': 'australian', 'en-CA': 'canadian', 'en-GB': 'british', 'en-NZ': 'newzealand', 'en-US': 'american', 'eo': 'esperanto', 'es': 'spanish', 'et': 'estonian', 'eu': 'basque', 'fi': 'finnish', 'fr': 'french', 'fr-CA': 'canadien', 'ga': 'irish', 'grc-ibycus': 'ibycus', 'gl': 'galician', 'he': 'hebrew', 'hr': 'croatian', 'hsb': 'uppersorbian', 'hu': 'magyar', 'ia': 'interlingua', 'id': 'bahasai', 'is': 'icelandic', 'it': 'italian', 'ja': 'japanese', 'kk': 'kazakh', 'la': 'latin', 'lt': 'lithuanian', 'lv': 'latvian', 'mn': 'mongolian', 'ms': 'bahasam', 'nb': 'norsk', 'nl': 'dutch', 'nn': 'nynorsk', 'no': 'norsk', 'pl': 'polish', 'pt': 'portuges', 'pt-BR': 'brazil', 'ro': 'romanian', 'ru': 'russian', 'se': 'samin', 'sh-Cyrl': 'serbianc', 'sh-Latn': 'serbian', 'sk': 'slovak', 'sl': 'slovene', 'sq': 'albanian', 'sr': 'serbianc', 'sr-Latn': 'serbian', 'sv': 'swedish', 'tr': 'turkish', 'uk': 'ukrainian', 'vi': 'vietnam'}
    language_codes = dict([(k.lower(), v) for k, v in list(language_codes.items())])
    warn_msg = 'Language "%s" not supported by LaTeX (babel)'
    active_chars = {'galician': '.<>', 'spanish': '.<>'}

    def __init__(self, language_code, reporter=None):
        self.reporter = reporter
        self.language = self.language_name(language_code)
        self.otherlanguages = {}

    def __call__(self):
        """Return the babel call with correct options and settings"""
        languages = sorted(self.otherlanguages.keys())
        languages.append(self.language or 'english')
        self.setup = ['\\usepackage[%s]{babel}' % ','.join(languages)]
        shorthands = []
        for c in ''.join([self.active_chars.get(l, '') for l in languages]):
            if c not in shorthands:
                shorthands.append(c)
        if shorthands:
            self.setup.append('\\AtBeginDocument{\\shorthandoff{%s}}' % ''.join(shorthands))
        if 'galician' in languages:
            self.setup.append('\\deactivatetilden % restore ~ in Galician')
        if 'estonian' in languages:
            self.setup.extend(['\\makeatletter', '  \\addto\\extrasestonian{\\bbl@deactivate{~}}', '\\makeatother'])
        if 'basque' in languages:
            self.setup.extend(['\\makeatletter', '  \\addto\\extrasbasque{\\bbl@deactivate{~}}', '\\makeatother'])
        if languages[-1] == 'english' and 'french' in list(self.otherlanguages.keys()):
            self.setup += ['% Prevent side-effects if French hyphenation patterns are not loaded:', '\\frenchbsetup{StandardLayout}', '\\AtBeginDocument{\\selectlanguage{%s}\\noextrasfrench}' % self.language]
        return '\n'.join(self.setup)

    def language_name(self, language_code):
        """Return TeX language name for `language_code`"""
        for tag in utils.normalize_language_tag(language_code):
            try:
                return self.language_codes[tag]
            except KeyError:
                pass
        if self.reporter is not None:
            self.reporter.warning(self.warn_msg % language_code)
        return ''

    def get_language(self):
        return self.language