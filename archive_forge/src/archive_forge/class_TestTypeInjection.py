from __future__ import absolute_import
import sys
import os.path
from textwrap import dedent
from contextlib import contextmanager
from tempfile import NamedTemporaryFile
from Cython.Compiler.ParseTreeTransforms import NormalizeTree, InterpretCompilerDirectives
from Cython.Compiler import Main, Symtab, Visitor, Options
from Cython.TestUtils import TransformTest
class TestTypeInjection(TestJediTyper):
    """
    Subtype of TestJediTyper that additionally tests type injection and compilation.
    """

    def setUp(self):
        super(TestTypeInjection, self).setUp()
        compilation_options = Options.CompilationOptions(Options.default_options)
        ctx = Main.Context.from_options(compilation_options)
        transform = InterpretCompilerDirectives(ctx, ctx.compiler_directives)
        transform.module_scope = Symtab.ModuleScope('__main__', None, ctx)
        self.declarations_finder = DeclarationsFinder()
        self.pipeline = [NormalizeTree(None), transform, self.declarations_finder]

    def _test(self, code):
        types, lines = _test_typing(code, inject=True)
        tree = self.run_pipeline(self.pipeline, ''.join(lines))
        directives = self.declarations_finder.directives
        return types