import re
from collections import namedtuple
class AppendParamDocumentation:
    """Appends documentation to a specific parameter"""

    def __init__(self, parameter_name, doc_string):
        self._parameter_name = parameter_name
        self._doc_string = doc_string

    def append_documentation(self, event_name, section, **kwargs):
        if self._parameter_name in section.available_sections:
            section = section.get_section(self._parameter_name)
            description_section = section.get_section('param-documentation')
            description_section.writeln(self._doc_string)