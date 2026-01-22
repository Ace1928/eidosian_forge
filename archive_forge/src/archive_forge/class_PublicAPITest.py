import unittest
import oslo_i18n
from oslo_i18n import _lazy
class PublicAPITest(unittest.TestCase):

    def test_create_factory(self):
        oslo_i18n.TranslatorFactory('domain')

    def test_install(self):
        oslo_i18n.install('domain')

    def test_get_available_languages(self):
        oslo_i18n.get_available_languages('domains')

    def test_toggle_lazy(self):
        original = _lazy.USE_LAZY
        try:
            oslo_i18n.enable_lazy(True)
            oslo_i18n.enable_lazy(False)
        finally:
            oslo_i18n.enable_lazy(original)

    def test_translate(self):
        oslo_i18n.translate('string')