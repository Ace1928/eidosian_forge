from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from fire import docstrings
from fire import testutils
def test_numpy_format_typed_args_and_returns(self):
    docstring = 'Docstring summary.\n\n    This is a longer description of the docstring. It spans across multiple\n    lines.\n\n    Parameters\n    ----------\n    param1 : int\n        The first parameter.\n    param2 : str\n        The second parameter.\n\n    Returns\n    -------\n    bool\n        True if successful, False otherwise.\n    '
    docstring_info = docstrings.parse(docstring)
    expected_docstring_info = DocstringInfo(summary='Docstring summary.', description='This is a longer description of the docstring. It spans across multiple\nlines.', args=[ArgInfo(name='param1', type='int', description='The first parameter.'), ArgInfo(name='param2', type='str', description='The second parameter.')], returns='bool True if successful, False otherwise.')
    self.assertEqual(expected_docstring_info, docstring_info)