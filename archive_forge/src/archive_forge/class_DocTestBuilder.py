import doctest
import re
import sys
import time
from io import StringIO
from os import path
from typing import (TYPE_CHECKING, Any, Callable, Dict, Iterable, List, Optional, Sequence,
from docutils import nodes
from docutils.nodes import Element, Node, TextElement
from docutils.parsers.rst import directives
from packaging.specifiers import InvalidSpecifier, SpecifierSet
from packaging.version import Version
import sphinx
from sphinx.builders import Builder
from sphinx.locale import __
from sphinx.util import logging
from sphinx.util.console import bold  # type: ignore
from sphinx.util.docutils import SphinxDirective
from sphinx.util.osutil import relpath
from sphinx.util.typing import OptionSpec
class DocTestBuilder(Builder):
    """
    Runs test snippets in the documentation.
    """
    name = 'doctest'
    epilog = __('Testing of doctests in the sources finished, look at the results in %(outdir)s/output.txt.')

    def init(self) -> None:
        self.opt = self.config.doctest_default_flags
        doctest.compile = self.compile
        sys.path[0:0] = self.config.doctest_path
        self.type = 'single'
        self.total_failures = 0
        self.total_tries = 0
        self.setup_failures = 0
        self.setup_tries = 0
        self.cleanup_failures = 0
        self.cleanup_tries = 0
        date = time.strftime('%Y-%m-%d %H:%M:%S')
        self.outfile = open(path.join(self.outdir, 'output.txt'), 'w', encoding='utf-8')
        self.outfile.write('Results of doctest builder run on %s\n==================================%s\n' % (date, '=' * len(date)))

    def _out(self, text: str) -> None:
        logger.info(text, nonl=True)
        self.outfile.write(text)

    def _warn_out(self, text: str) -> None:
        if self.app.quiet or self.app.warningiserror:
            logger.warning(text)
        else:
            logger.info(text, nonl=True)
        self.outfile.write(text)

    def get_target_uri(self, docname: str, typ: Optional[str]=None) -> str:
        return ''

    def get_outdated_docs(self) -> Set[str]:
        return self.env.found_docs

    def finish(self) -> None:

        def s(v: int) -> str:
            return 's' if v != 1 else ''
        repl = (self.total_tries, s(self.total_tries), self.total_failures, s(self.total_failures), self.setup_failures, s(self.setup_failures), self.cleanup_failures, s(self.cleanup_failures))
        self._out('\nDoctest summary\n===============\n%5d test%s\n%5d failure%s in tests\n%5d failure%s in setup code\n%5d failure%s in cleanup code\n' % repl)
        self.outfile.close()
        if self.total_failures or self.setup_failures or self.cleanup_failures:
            self.app.statuscode = 1

    def write(self, build_docnames: Iterable[str], updated_docnames: Sequence[str], method: str='update') -> None:
        if build_docnames is None:
            build_docnames = sorted(self.env.all_docs)
        logger.info(bold('running tests...'))
        for docname in build_docnames:
            doctree = self.env.get_doctree(docname)
            self.test_doc(docname, doctree)

    def get_filename_for_node(self, node: Node, docname: str) -> str:
        """Try to get the file which actually contains the doctest, not the
        filename of the document it's included in."""
        try:
            filename = relpath(node.source, self.env.srcdir).rsplit(':docstring of ', maxsplit=1)[0]
        except Exception:
            filename = self.env.doc2path(docname, False)
        return filename

    @staticmethod
    def get_line_number(node: Node) -> Optional[int]:
        """Get the real line number or admit we don't know."""
        if ':docstring of ' in path.basename(node.source or ''):
            return None
        if node.line is not None:
            return node.line - 1
        return None

    def skipped(self, node: Element) -> bool:
        if 'skipif' not in node:
            return False
        else:
            condition = node['skipif']
            context: Dict[str, Any] = {}
            if self.config.doctest_global_setup:
                exec(self.config.doctest_global_setup, context)
            should_skip = eval(condition, context)
            if self.config.doctest_global_cleanup:
                exec(self.config.doctest_global_cleanup, context)
            return should_skip

    def test_doc(self, docname: str, doctree: Node) -> None:
        groups: Dict[str, TestGroup] = {}
        add_to_all_groups = []
        self.setup_runner = SphinxDocTestRunner(verbose=False, optionflags=self.opt)
        self.test_runner = SphinxDocTestRunner(verbose=False, optionflags=self.opt)
        self.cleanup_runner = SphinxDocTestRunner(verbose=False, optionflags=self.opt)
        self.test_runner._fakeout = self.setup_runner._fakeout
        self.cleanup_runner._fakeout = self.setup_runner._fakeout
        if self.config.doctest_test_doctest_blocks:

            def condition(node: Node) -> bool:
                return isinstance(node, (nodes.literal_block, nodes.comment)) and 'testnodetype' in node or isinstance(node, nodes.doctest_block)
        else:

            def condition(node: Node) -> bool:
                return isinstance(node, (nodes.literal_block, nodes.comment)) and 'testnodetype' in node
        for node in doctree.findall(condition):
            if self.skipped(node):
                continue
            source = node['test'] if 'test' in node else node.astext()
            filename = self.get_filename_for_node(node, docname)
            line_number = self.get_line_number(node)
            if not source:
                logger.warning(__('no code/output in %s block at %s:%s'), node.get('testnodetype', 'doctest'), filename, line_number)
            code = TestCode(source, type=node.get('testnodetype', 'doctest'), filename=filename, lineno=line_number, options=node.get('options'))
            node_groups = node.get('groups', ['default'])
            if '*' in node_groups:
                add_to_all_groups.append(code)
                continue
            for groupname in node_groups:
                if groupname not in groups:
                    groups[groupname] = TestGroup(groupname)
                groups[groupname].add_code(code)
        for code in add_to_all_groups:
            for group in groups.values():
                group.add_code(code)
        if self.config.doctest_global_setup:
            code = TestCode(self.config.doctest_global_setup, 'testsetup', filename=None, lineno=0)
            for group in groups.values():
                group.add_code(code, prepend=True)
        if self.config.doctest_global_cleanup:
            code = TestCode(self.config.doctest_global_cleanup, 'testcleanup', filename=None, lineno=0)
            for group in groups.values():
                group.add_code(code)
        if not groups:
            return
        self._out('\nDocument: %s\n----------%s\n' % (docname, '-' * len(docname)))
        for group in groups.values():
            self.test_group(group)
        res_f, res_t = self.setup_runner.summarize(self._out, verbose=False)
        self.setup_failures += res_f
        self.setup_tries += res_t
        if self.test_runner.tries:
            res_f, res_t = self.test_runner.summarize(self._out, verbose=True)
            self.total_failures += res_f
            self.total_tries += res_t
        if self.cleanup_runner.tries:
            res_f, res_t = self.cleanup_runner.summarize(self._out, verbose=True)
            self.cleanup_failures += res_f
            self.cleanup_tries += res_t

    def compile(self, code: str, name: str, type: str, flags: Any, dont_inherit: bool) -> Any:
        return compile(code, name, self.type, flags, dont_inherit)

    def test_group(self, group: TestGroup) -> None:
        ns: Dict = {}

        def run_setup_cleanup(runner: Any, testcodes: List[TestCode], what: Any) -> bool:
            examples = []
            for testcode in testcodes:
                example = doctest.Example(testcode.code, '', lineno=testcode.lineno)
                examples.append(example)
            if not examples:
                return True
            sim_doctest = doctest.DocTest(examples, {}, '%s (%s code)' % (group.name, what), testcodes[0].filename, 0, None)
            sim_doctest.globs = ns
            old_f = runner.failures
            self.type = 'exec'
            runner.run(sim_doctest, out=self._warn_out, clear_globs=False)
            if runner.failures > old_f:
                return False
            return True
        if not run_setup_cleanup(self.setup_runner, group.setup, 'setup'):
            return
        for code in group.tests:
            if len(code) == 1:
                try:
                    test = parser.get_doctest(code[0].code, {}, group.name, code[0].filename, code[0].lineno)
                except Exception:
                    logger.warning(__('ignoring invalid doctest code: %r'), code[0].code, location=(code[0].filename, code[0].lineno))
                    continue
                if not test.examples:
                    continue
                for example in test.examples:
                    new_opt = code[0].options.copy()
                    new_opt.update(example.options)
                    example.options = new_opt
                self.type = 'single'
            else:
                output = code[1].code if code[1] else ''
                options = code[1].options if code[1] else {}
                options[doctest.DONT_ACCEPT_BLANKLINE] = True
                m = parser._EXCEPTION_RE.match(output)
                if m:
                    exc_msg = m.group('msg')
                else:
                    exc_msg = None
                example = doctest.Example(code[0].code, output, exc_msg=exc_msg, lineno=code[0].lineno, options=options)
                test = doctest.DocTest([example], {}, group.name, code[0].filename, code[0].lineno, None)
                self.type = 'exec'
            test.globs = ns
            self.test_runner.run(test, out=self._warn_out, clear_globs=False)
        run_setup_cleanup(self.cleanup_runner, group.cleanup, 'cleanup')