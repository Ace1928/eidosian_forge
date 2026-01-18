from pyflakes import messages as m
from pyflakes.checker import (FunctionScope, ClassScope, ModuleScope,
from pyflakes.test.harness import TestCase
def test_scope_class(self):
    checker = self.flakes('\n        class Foo:\n            x = 0\n            def bar(a, b=1, *d, **e):\n                pass\n        ', is_segment=True)
    scopes = checker.deadScopes
    module_scopes = [scope for scope in scopes if scope.__class__ is ModuleScope]
    class_scopes = [scope for scope in scopes if scope.__class__ is ClassScope]
    function_scopes = [scope for scope in scopes if scope.__class__ is FunctionScope]
    self.assertEqual(len(module_scopes), 0)
    self.assertEqual(len(class_scopes), 1)
    self.assertEqual(len(function_scopes), 1)
    class_scope = class_scopes[0]
    function_scope = function_scopes[0]
    self.assertIsInstance(class_scope, ClassScope)
    self.assertIsInstance(function_scope, FunctionScope)
    self.assertIn('x', class_scope)
    self.assertIn('bar', class_scope)
    self.assertIn('a', function_scope)
    self.assertIn('b', function_scope)
    self.assertIn('d', function_scope)
    self.assertIn('e', function_scope)
    self.assertIsInstance(class_scope['bar'], FunctionDefinition)
    self.assertIsInstance(class_scope['x'], Assignment)
    self.assertIsInstance(function_scope['a'], Argument)
    self.assertIsInstance(function_scope['b'], Argument)
    self.assertIsInstance(function_scope['d'], Argument)
    self.assertIsInstance(function_scope['e'], Argument)