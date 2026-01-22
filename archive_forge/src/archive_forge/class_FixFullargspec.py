from lib2to3 import fixer_base
from lib2to3.fixer_util import Name
class FixFullargspec(fixer_base.BaseFix):
    PATTERN = u"'getfullargspec'"

    def transform(self, node, results):
        self.warning(node, warn_msg)
        return Name(u'getargspec', prefix=node.prefix)