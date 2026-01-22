import numbers
import re
from botocore.docs.utils import escape_controls
from botocore.utils import parse_timestamp
def _document_list(self, section, value, comments, path, shape):
    list_section = section.add_new_section('list-section')
    self._start_nested_value(list_section, '[')
    item_shape = shape.member
    for index, val in enumerate(value):
        item_section = list_section.add_new_section(index)
        item_section.style.new_line()
        path.append('[%s]' % index)
        item_comment = self._get_comment(path, comments)
        if item_comment:
            item_section.write(item_comment)
            item_section.style.new_line()
        self._document(item_section, val, comments, path, item_shape)
        path.pop()
    list_section_end = list_section.add_new_section('ending-bracket')
    self._end_nested_value(list_section_end, '],')