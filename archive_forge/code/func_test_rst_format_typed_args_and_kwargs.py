from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from fire import docstrings
from fire import testutils
def test_rst_format_typed_args_and_kwargs(self):
    docstring = 'Docstring summary.\n\n    :param arg1: Description of arg1.\n    :type arg1: str.\n    :key arg2: Description of arg2.\n    :type arg2: bool.\n    :key arg3: Description of arg3.\n    :type arg3: str.\n    '
    docstring_info = docstrings.parse(docstring)
    expected_docstring_info = DocstringInfo(summary='Docstring summary.', args=[ArgInfo(name='arg1', type='str', description='Description of arg1.'), KwargInfo(name='arg2', type='bool', description='Description of arg2.'), KwargInfo(name='arg3', type='str', description='Description of arg3.')])
    self.assertEqual(expected_docstring_info, docstring_info)