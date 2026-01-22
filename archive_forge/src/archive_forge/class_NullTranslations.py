import operator
import os
import re
import sys
class NullTranslations:

    def __init__(self, fp=None):
        self._info = {}
        self._charset = None
        self._fallback = None
        if fp is not None:
            self._parse(fp)

    def _parse(self, fp):
        pass

    def add_fallback(self, fallback):
        if self._fallback:
            self._fallback.add_fallback(fallback)
        else:
            self._fallback = fallback

    def gettext(self, message):
        if self._fallback:
            return self._fallback.gettext(message)
        return message

    def ngettext(self, msgid1, msgid2, n):
        if self._fallback:
            return self._fallback.ngettext(msgid1, msgid2, n)
        if n == 1:
            return msgid1
        else:
            return msgid2

    def pgettext(self, context, message):
        if self._fallback:
            return self._fallback.pgettext(context, message)
        return message

    def npgettext(self, context, msgid1, msgid2, n):
        if self._fallback:
            return self._fallback.npgettext(context, msgid1, msgid2, n)
        if n == 1:
            return msgid1
        else:
            return msgid2

    def info(self):
        return self._info

    def charset(self):
        return self._charset

    def install(self, names=None):
        import builtins
        builtins.__dict__['_'] = self.gettext
        if names is not None:
            allowed = {'gettext', 'ngettext', 'npgettext', 'pgettext'}
            for name in allowed & set(names):
                builtins.__dict__[name] = getattr(self, name)