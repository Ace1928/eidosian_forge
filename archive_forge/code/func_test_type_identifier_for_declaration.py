from __future__ import absolute_import
import unittest
import Cython.Compiler.PyrexTypes as PT
def test_type_identifier_for_declaration(self):
    test_func = PT.type_identifier_from_declaration
    function_name = test_func.__name__
    self._test_escape(function_name)
    test_data = [('const &std::vector', 'const__refstd__in_vector'), ('const &std::vector<int>', 'const__refstd__in_vector__lAngint__rAng'), ('const &&std::vector', 'const__fwrefstd__in_vector'), ('const &&&std::vector', 'const__fwref__refstd__in_vector'), ('const &&std::vector', 'const__fwrefstd__in_vector'), ('void (*func)(int x, float y)', '975d51__void__lParen__ptrfunc__rParen__lParenint__spac__etc'), ('float ** (*func)(int x, int[:] y)', '31883a__float__ptr__ptr__lParen__ptrfunc__rParen__lPar__etc')]
    self._test_escape(function_name, test_data)