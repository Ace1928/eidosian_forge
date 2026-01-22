from botocore.docs.shape import ShapeDocumenter
from botocore.docs.utils import py_type_name
class RequestParamsDocumenter(BaseParamsDocumenter):
    """Generates the description for the request parameters"""
    EVENT_NAME = 'request-params'

    def document_shape_type_structure(self, section, shape, history, include=None, exclude=None, **kwargs):
        if len(history) > 1:
            self._add_member_documentation(section, shape, **kwargs)
            section.style.indent()
        members = self._add_members_to_shape(shape.members, include)
        for i, param in enumerate(members):
            if exclude and param in exclude:
                continue
            param_shape = members[param]
            param_section = section.add_new_section(param, context={'shape': param_shape.name})
            param_section.style.new_line()
            is_required = param in shape.required_members
            self.traverse_and_document_shape(section=param_section, shape=param_shape, history=history, name=param, is_required=is_required)
        section = section.add_new_section('end-structure')
        if len(history) > 1:
            section.style.dedent()
        section.style.new_line()

    def _add_member_documentation(self, section, shape, name=None, is_top_level_param=False, is_required=False, **kwargs):
        py_type = self._get_special_py_type_name(shape)
        if py_type is None:
            py_type = py_type_name(shape.type_name)
        if is_top_level_param:
            type_section = section.add_new_section('param-type')
            type_section.write(f':type {name}: {py_type}')
            end_type_section = type_section.add_new_section('end-param-type')
            end_type_section.style.new_line()
            name_section = section.add_new_section('param-name')
            name_section.write(':param %s: ' % name)
        else:
            name_section = section.add_new_section('param-name')
            name_section.write('- ')
            if name is not None:
                name_section.style.bold('%s' % name)
                name_section.write(' ')
            type_section = section.add_new_section('param-type')
            self._document_non_top_level_param_type(type_section, shape)
        if is_required:
            is_required_section = section.add_new_section('is-required')
            is_required_section.style.indent()
            is_required_section.style.bold('[REQUIRED]')
            is_required_section.write(' ')
        if shape.documentation:
            documentation_section = section.add_new_section('param-documentation')
            documentation_section.style.indent()
            if getattr(shape, 'is_tagged_union', False):
                tagged_union_docs = section.add_new_section('param-tagged-union-docs')
                note = '.. note::    This is a Tagged Union structure. Only one of the     following top level keys can be set: %s. '
                tagged_union_members_str = ', '.join(['``%s``' % key for key in shape.members.keys()])
                tagged_union_docs.write(note % tagged_union_members_str)
            documentation_section.include_doc_string(shape.documentation)
            self._add_special_trait_documentation(documentation_section, shape)
        end_param_section = section.add_new_section('end-param')
        end_param_section.style.new_paragraph()

    def _add_special_trait_documentation(self, section, shape):
        if 'idempotencyToken' in shape.metadata:
            self._append_idempotency_documentation(section)

    def _append_idempotency_documentation(self, section):
        docstring = 'This field is autopopulated if not provided.'
        section.write(docstring)