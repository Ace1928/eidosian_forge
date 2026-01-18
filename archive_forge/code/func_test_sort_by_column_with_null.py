import weakref
from unittest import mock
from cliff import lister
from cliff.tests import base
def test_sort_by_column_with_null(self):
    test_lister = ExerciseListerNullValues(mock.Mock(), [])
    parsed_args = mock.Mock()
    parsed_args.columns = ('Col1', 'Col2')
    parsed_args.formatter = 'test'
    parsed_args.sort_columns = ['Col2', 'Col1']
    test_lister.run(parsed_args)
    f = test_lister._formatter_plugins['test']
    args = f.args[0]
    data = list(args[1])
    self.assertEqual([['a', 'A'], ['c', 'A'], ['b', 'B'], [None, None]], data)