import weakref
from unittest import mock
from cliff import lister
from cliff.tests import base
def test_sort_by_column_with_different_types(self):
    test_lister = ExerciseListerDifferentTypes(mock.Mock(), [])
    parsed_args = mock.Mock()
    parsed_args.columns = ('Col1', 'Col2')
    parsed_args.formatter = 'test'
    parsed_args.sort_columns = ['Col2', 'Col1']
    with mock.patch.object(lister.Lister, 'log') as mock_log:
        test_lister.run(parsed_args)
    f = test_lister._formatter_plugins['test']
    args = f.args[0]
    data = list(args[1])
    self.assertEqual([['a', 'A'], ['b', 'B'], ['c', 'A'], [1, 0]], data)
    mock_log.warning.assert_has_calls([mock.call("Could not sort on field '%s'; unsortable types", col) for col in parsed_args.sort_columns])