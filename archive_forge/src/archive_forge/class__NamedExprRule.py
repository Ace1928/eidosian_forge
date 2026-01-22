import codecs
import warnings
import re
from contextlib import contextmanager
from parso.normalizer import Normalizer, NormalizerConfig, Issue, Rule
from parso.python.tokenize import _get_token_collection
@ErrorFinder.register_rule(type='namedexpr_test')
class _NamedExprRule(_CheckAssignmentRule):

    def is_issue(self, namedexpr_test):
        first = namedexpr_test.children[0]

        def search_namedexpr_in_comp_for(node):
            while True:
                parent = node.parent
                if parent is None:
                    return parent
                if parent.type == 'sync_comp_for' and parent.children[3] == node:
                    return parent
                node = parent
        if search_namedexpr_in_comp_for(namedexpr_test):
            message = 'assignment expression cannot be used in a comprehension iterable expression'
            self.add_issue(namedexpr_test, message=message)
        exprlist = list()

        def process_comp_for(comp_for):
            if comp_for.type == 'sync_comp_for':
                comp = comp_for
            elif comp_for.type == 'comp_for':
                comp = comp_for.children[1]
            exprlist.extend(_get_for_stmt_definition_exprs(comp))

        def search_all_comp_ancestors(node):
            has_ancestors = False
            while True:
                node = node.search_ancestor('testlist_comp', 'dictorsetmaker')
                if node is None:
                    break
                for child in node.children:
                    if child.type in _COMP_FOR_TYPES:
                        process_comp_for(child)
                        has_ancestors = True
                        break
            return has_ancestors
        search_all = search_all_comp_ancestors(namedexpr_test)
        if search_all:
            if self._normalizer.context.node.type == 'classdef':
                message = 'assignment expression within a comprehension cannot be used in a class body'
                self.add_issue(namedexpr_test, message=message)
            namelist = [expr.value for expr in exprlist if expr.type == 'name']
            if first.type == 'name' and first.value in namelist:
                message = 'assignment expression cannot rebind comprehension iteration variable %r' % first.value
                self.add_issue(namedexpr_test, message=message)
        self._check_assignment(first, is_namedexpr=True)