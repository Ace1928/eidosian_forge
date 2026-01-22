import builtins
import gettext
import logging
from unittest import mock
from oslotest import base as test_base
from oslo_i18n import _factory
from oslo_i18n import _gettextutils
from oslo_i18n import _lazy
from oslo_i18n import _message
def _mock_gettext_find(domain, localedir=None, languages=None, all=0):
    languages = languages or []
    if domain == 'domain_1':
        if any((x in ['en_GB', 'es_ES', 'fil_PH', 'it'] for x in languages)):
            return 'translation-file'
    elif domain == 'domain_2':
        if any((x in ['fr_FR', 'zh_HK'] for x in languages)):
            return 'translation-file'
    return None