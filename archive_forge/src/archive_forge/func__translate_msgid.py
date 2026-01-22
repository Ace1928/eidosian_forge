import copy
import gettext
import locale
import logging
import os
import warnings
from oslo_i18n import _locale
from oslo_i18n import _translate
@staticmethod
def _translate_msgid(msgid, domain, desired_locale=None, has_contextual_form=False, has_plural_form=False):
    if not desired_locale:
        system_locale = locale.getlocale(locale.LC_CTYPE)
        if not system_locale or not system_locale[0]:
            desired_locale = 'en_US'
        else:
            desired_locale = system_locale[0]
    locale_dir = os.environ.get(_locale.get_locale_dir_variable_name(domain))
    lang = gettext.translation(domain, localedir=locale_dir, languages=[desired_locale], fallback=True)
    if not has_contextual_form and (not has_plural_form):
        translator = lang.gettext
        translated_message = translator(msgid)
    elif has_contextual_form and has_plural_form:
        raise ValueError('Unimplemented.')
    elif has_contextual_form:
        msgctx, msgtxt = msgid
        translator = lang.gettext
        msg_with_ctx = '%s%s%s' % (msgctx, CONTEXT_SEPARATOR, msgtxt)
        translated_message = translator(msg_with_ctx)
        if CONTEXT_SEPARATOR in translated_message:
            translated_message = msgtxt
    elif has_plural_form:
        msgsingle, msgplural, msgcount = msgid
        translator = lang.ngettext
        translated_message = translator(msgsingle, msgplural, msgcount)
    return translated_message