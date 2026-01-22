from lib2to3.fixer_base import BaseFix
from lib2to3.fixer_util import Name, Assign
class FixStdDev(BaseFix):
    PATTERN = "\n    power< any* trailer< '.' 'std_dev' > trailer< '(' ')' > >\n    |\n    power< any* trailer< '.' 'set_std_dev' > trailer< '(' set_arg=any ')' > >\n    "

    def transform(self, node, results):
        if 'set_arg' in results:
            attribute = node.children[-2]
            attribute.children[1].replace(Name('std_dev'))
            node.children[-1].remove()
            node.replace(Assign(node.clone(), results['set_arg'].clone()))
        else:
            node.children[-1].remove()