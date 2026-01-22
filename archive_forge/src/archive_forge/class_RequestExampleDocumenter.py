from botocore.docs.shape import ShapeDocumenter
from botocore.docs.utils import py_default
class RequestExampleDocumenter(BaseExampleDocumenter):
    EVENT_NAME = 'request-example'

    def document_shape_type_structure(self, section, shape, history, include=None, exclude=None, **kwargs):
        param_format = "'%s'"
        operator = ': '
        start = '{'
        end = '}'
        if len(history) <= 1:
            operator = '='
            start = '('
            end = ')'
            param_format = '%s'
        section = section.add_new_section('structure-value')
        self._start_nested_param(section, start)
        input_members = self._add_members_to_shape(shape.members, include)
        for i, param in enumerate(input_members):
            if exclude and param in exclude:
                continue
            param_section = section.add_new_section(param)
            param_section.write(param_format % param)
            param_section.write(operator)
            param_shape = input_members[param]
            param_value_section = param_section.add_new_section('member-value', context={'shape': param_shape.name})
            self.traverse_and_document_shape(section=param_value_section, shape=param_shape, history=history, name=param)
            if i < len(input_members) - 1:
                ending_comma_section = param_section.add_new_section('ending-comma')
                ending_comma_section.write(',')
                ending_comma_section.style.new_line()
        self._end_structure(section, start, end)