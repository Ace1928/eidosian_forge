import numbers
import re
from botocore.docs.utils import escape_controls
from botocore.utils import parse_timestamp
def _document_params(self, section, value, comments, path, shape):
    param_section = section.add_new_section('param-values')
    self._start_nested_value(param_section, '(')
    for key, val in value.items():
        path.append('.%s' % key)
        item_section = param_section.add_new_section(key)
        item_section.style.new_line()
        item_comment = self._get_comment(path, comments)
        if item_comment:
            item_section.write(item_comment)
            item_section.style.new_line()
        item_section.write(key + '=')
        item_shape = None
        if shape:
            item_shape = shape.members.get(key)
        self._document(item_section, val, comments, path, item_shape)
        path.pop()
    param_section_end = param_section.add_new_section('ending-parenthesis')
    self._end_nested_value(param_section_end, ')')