from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from fire import docstrings
from fire import testutils
def test_rst_format_typed_args_and_returns(self):
    docstring = 'Docstring summary.\n\n    This is a longer description of the docstring. It spans across multiple\n    lines.\n\n    :param arg1: Description of arg1.\n    :type arg1: str.\n    :param arg2: Description of arg2.\n    :type arg2: bool.\n    :returns:  int -- description of the return value.\n    :raises: AttributeError, KeyError\n    '
    docstring_info = docstrings.parse(docstring)
    expected_docstring_info = DocstringInfo(summary='Docstring summary.', description='This is a longer description of the docstring. It spans across multiple\nlines.', args=[ArgInfo(name='arg1', type='str', description='Description of arg1.'), ArgInfo(name='arg2', type='bool', description='Description of arg2.')], returns='int -- description of the return value.', raises='AttributeError, KeyError')
    self.assertEqual(expected_docstring_info, docstring_info)