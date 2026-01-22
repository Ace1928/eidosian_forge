import gettext
def _translation(domain, localedir=None, languages=None, fallback=None):
    if languages:
        language = languages[0]
        if language in locales_map:
            return FakeTranslations(locales_map[language])
    return gettext.NullTranslations()