import re
from collections import namedtuple
def append_documentation(self, event_name, section, **kwargs):
    if self._parameter_name in section.available_sections:
        section = section.get_section(self._parameter_name)
        description_section = section.get_section('param-documentation')
        description_section.writeln(self._doc_string)