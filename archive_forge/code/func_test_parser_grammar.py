import io
import sys
import yaql
from yaql.language import exceptions
from yaql.language import factory
from yaql.language import specs
from yaql.language import yaqltypes
from yaql import tests
def test_parser_grammar(self):
    copy = sys.stderr
    sys.stderr = io.StringIO()
    try:
        debug_opts = dict(self.engine_options)
        debug_opts['yaql.debug'] = True
        yaql.factory.YaqlFactory().create(options=debug_opts)
        sys.stderr.seek(0)
        err_out = sys.stderr.read()
        self.assertEqual('Generating LALR tables\n', err_out)
    finally:
        sys.stderr = copy