from lib2to3.fixer_base import BaseFix
from lib2to3.fixer_util import ArgList, Call, Comma, Name, syms
class FixUfloat(BaseFix):
    PATTERN = "\n        power< 'ufloat' {tuple_call} any* >\n        |\n        power< 'ufloat' {tuple_any_call} any* >\n        |\n        power< 'ufloat' trailer< '(' string=STRING ')' > any* >\n        |\n        power< 'ufloat' trailer< '('\n            arglist<\n                string=STRING\n                ',' tag=any\n            >\n        ')' > any* >\n        |\n        power< object=NAME trailer< '.' 'ufloat' > {tuple_call} any* >\n        |\n        power< object=NAME trailer< '.' 'ufloat' > {tuple_any_call} any* >\n        |\n        power< object=NAME trailer< '.' 'ufloat' >\n        trailer< '(' string=STRING ')' >\n        any* >\n        |\n        power< object=NAME trailer< '.' 'ufloat' >\n        trailer< '(' arglist< string=STRING ',' tag=any > ')' >\n        any* >\n        ".format(tuple_call=tuple_call, tuple_any_call=tuple_any_call)

    def transform(self, node, results):
        if 'string' in results:
            new_func_name = 'ufloat_fromstr'
            new_args = [results['string'].clone()]
        else:
            new_func_name = 'ufloat'
            new_args = [results['arg0'].clone(), Comma(), results['arg1'].clone()]
        if 'tag' in results:
            new_args.extend([Comma(), results['tag'].clone()])
        if 'object' in results:
            func_name = node.children[1].children[1]
            args = node.children[2]
        else:
            func_name = node.children[0]
            args = node.children[1]
        func_name.value = new_func_name
        args.replace(ArgList(new_args))