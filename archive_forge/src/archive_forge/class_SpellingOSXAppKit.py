from AppKit import NSSpellChecker, NSMakeRange
from kivy.core.spelling import SpellingBase, NoSuchLangError
class SpellingOSXAppKit(SpellingBase):
    """
    Spelling backend based on OSX's spelling features provided by AppKit.
    """

    def __init__(self, language=None):
        self._language = NSSpellChecker.alloc().init()
        super(SpellingOSXAppKit, self).__init__(language)

    def select_language(self, language):
        success = self._language.setLanguage_(language)
        if not success:
            err = 'AppKit Backend: No language "%s" ' % (language,)
            raise NoSuchLangError(err)

    def list_languages(self):
        return list(self._language.availableLanguages())

    def check(self, word):
        if not word:
            return None
        err = 'check() not currently supported by the OSX AppKit backend'
        raise NotImplementedError(err)

    def suggest(self, fragment):
        l = self._language
        try:
            return list(l.guessesForWord_(fragment))
        except AttributeError:
            checkrange = NSMakeRange(0, len(fragment))
            g = l.guessesForWordRange_inString_language_inSpellDocumentWithTag_(checkrange, fragment, l.language(), 0)
            return list(g)