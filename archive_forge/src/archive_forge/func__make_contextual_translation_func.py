import gettext
import os
from oslo_i18n import _lazy
from oslo_i18n import _locale
from oslo_i18n import _message
def _make_contextual_translation_func(self, domain=None):
    """Return a translation function ready for use with context messages.

        The returned function takes two values, the context of
        the unicode string, the unicode string to be translated.
        The returned type is the same as
        :method:`TranslatorFactory._make_translation_func`.

        The domain argument is the same as
        :method:`TranslatorFactory._make_translation_func`.

        """
    if domain is None:
        domain = self.domain
    t = gettext.translation(domain, localedir=self.localedir, fallback=True)
    m = t.gettext

    def f(ctx, msg):
        """oslo.i18n.gettextutils translation with context function."""
        if _lazy.USE_LAZY:
            msgid = (ctx, msg)
            return _message.Message(msgid, domain=domain, has_contextual_form=True)
        msgctx = '%s%s%s' % (ctx, CONTEXT_SEPARATOR, msg)
        s = m(msgctx)
        if CONTEXT_SEPARATOR in s:
            return msg
        return s
    return f