from lib2to3.fixer_base import BaseFix
from lib2to3.fixer_util import String, ArgList, Comma, syms
class FixUarrayUmatrix(BaseFix):
    PATTERN = "\n        power< 'uarray' {tuple_call} any* >\n        |\n        power< object=NAME trailer< '.' 'uarray' > {tuple_call} any* >\n        |\n        power< 'uarray' trailer< '(' args=any ')' > any* >\n        |\n        power< object=NAME trailer< '.' 'uarray' >\n            trailer< '(' args=any ')' >\n        any* >\n        ".format(tuple_call=tuple_call)
    PATTERN = '{}|{}'.format(PATTERN, PATTERN.replace('uarray', 'umatrix'))

    def transform(self, node, results):
        if 'object' in results:
            args = node.children[2]
        else:
            args = node.children[1]
        if 'args' in results:
            args_node = results['args']
            if args_node.type == syms.arglist:
                return
            new_args = [String('*'), args.children[1].clone()]
        else:
            new_args = [results['arg0'].clone(), Comma(), results['arg1'].clone()]
        args.replace(ArgList(new_args))