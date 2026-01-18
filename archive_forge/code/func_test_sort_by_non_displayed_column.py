import weakref
from unittest import mock
from cliff import lister
from cliff.tests import base
def test_sort_by_non_displayed_column(self):
    test_lister = ExerciseLister(mock.Mock(), [])
    parsed_args = mock.Mock()
    parsed_args.columns = ('Col1',)
    parsed_args.formatter = 'test'
    parsed_args.sort_columns = ['Col2']
    with mock.patch.object(test_lister, 'take_action') as mock_take_action:
        mock_take_action.return_value = (('Col1', 'Col2'), [['a', 'A'], ['b', 'B'], ['c', 'A']])
        test_lister.run(parsed_args)
    f = test_lister._formatter_plugins['test']
    args = f.args[0]
    data = list(args[1])
    self.assertEqual([['a'], ['c'], ['b']], data)