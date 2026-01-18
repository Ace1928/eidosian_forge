import io
from .. import errors, i18n, tests, workingtree
def test_translation(self):
    trans = ZzzTranslations()
    t = trans.zzz('msg')
    self._check_exact('zzå{{msg}}', t)
    t = trans.gettext('msg')
    self._check_exact('zzå{{msg}}', t)
    t = trans.ngettext('msg1', 'msg2', 0)
    self._check_exact('zzå{{msg2}}', t)
    t = trans.ngettext('msg1', 'msg2', 2)
    self._check_exact('zzå{{msg2}}', t)
    t = trans.ngettext('msg1', 'msg2', 1)
    self._check_exact('zzå{{msg1}}', t)