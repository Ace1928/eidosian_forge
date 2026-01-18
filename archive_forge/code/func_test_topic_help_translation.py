import io
from .. import errors, i18n, tests, workingtree
def test_topic_help_translation(self):
    """does topic help get translated?"""
    from .. import help
    out = io.StringIO()
    help.help('authentication', out)
    self.assertContainsRe(out.getvalue(), 'zzå{{Authentication Settings')