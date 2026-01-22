import inspect
import jmespath
from botocore.compat import six
class DocumentModifiedShape(object):

    def __init__(self, shape_name, new_type, new_description, new_example_value):
        self._shape_name = shape_name
        self._new_type = new_type
        self._new_description = new_description
        self._new_example_value = new_example_value

    def replace_documentation_for_matching_shape(self, event_name, section, **kwargs):
        if self._shape_name == section.context.get('shape'):
            self._replace_documentation(event_name, section)
        for section_name in section.available_sections:
            sub_section = section.get_section(section_name)
            if self._shape_name == sub_section.context.get('shape'):
                self._replace_documentation(event_name, sub_section)
            else:
                self.replace_documentation_for_matching_shape(event_name, sub_section)

    def _replace_documentation(self, event_name, section):
        if event_name.startswith('docs.request-example') or event_name.startswith('docs.response-example'):
            section.remove_all_sections()
            section.clear_text()
            section.write(self._new_example_value)
        if event_name.startswith('docs.request-params') or event_name.startswith('docs.response-params'):
            for section_name in section.available_sections:
                if section_name not in ['param-name', 'param-documentation', 'end-structure', 'param-type', 'end-param']:
                    section.delete_section(section_name)
            description_section = section.get_section('param-documentation')
            description_section.clear_text()
            description_section.write(self._new_description)
            type_section = section.get_section('param-type')
            if type_section.getvalue().decode('utf-8').startswith(':type'):
                type_section.clear_text()
                type_section.write(':type %s: %s' % (section.name, self._new_type))
            else:
                type_section.clear_text()
                type_section.style.italics('(%s) -- ' % self._new_type)