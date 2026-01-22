from typing import TYPE_CHECKING, Any, Dict, List, Tuple, Type, Union, cast
from docutils import nodes
from docutils.nodes import Node
from docutils.parsers.rst.states import Inliner
from sphinx import addnodes
from sphinx.environment import BuildEnvironment
from sphinx.locale import __
from sphinx.util import logging
from sphinx.util.typing import TextlikeNode
class DocFieldTransformer:
    """
    Transforms field lists in "doc field" syntax into better-looking
    equivalents, using the field type definitions given on a domain.
    """
    typemap: Dict[str, Tuple[Field, bool]]

    def __init__(self, directive: 'ObjectDescription') -> None:
        self.directive = directive
        self.typemap = directive.get_field_type_map()

    def transform_all(self, node: addnodes.desc_content) -> None:
        """Transform all field list children of a node."""
        for child in node:
            if isinstance(child, nodes.field_list):
                self.transform(child)

    def transform(self, node: nodes.field_list) -> None:
        """Transform a single field list *node*."""
        typemap = self.typemap
        entries: List[Union[nodes.field, Tuple[Field, Any, Node]]] = []
        groupindices: Dict[str, int] = {}
        types: Dict[str, Dict] = {}
        for field in cast(List[nodes.field], node):
            assert len(field) == 2
            field_name = cast(nodes.field_name, field[0])
            field_body = cast(nodes.field_body, field[1])
            try:
                fieldtype_name, fieldarg = field_name.astext().split(None, 1)
            except ValueError:
                fieldtype_name, fieldarg = (field_name.astext(), '')
            typedesc, is_typefield = typemap.get(fieldtype_name, (None, None))
            if _is_single_paragraph(field_body):
                paragraph = cast(nodes.paragraph, field_body[0])
                content = paragraph.children
            else:
                content = field_body.children
            if typedesc is None or typedesc.has_arg != bool(fieldarg):
                new_fieldname = fieldtype_name[0:1].upper() + fieldtype_name[1:]
                if fieldarg:
                    new_fieldname += ' ' + fieldarg
                field_name[0] = nodes.Text(new_fieldname)
                entries.append(field)
                if typedesc and is_typefield and content and (len(content) == 1) and isinstance(content[0], nodes.Text):
                    typed_field = cast(TypedField, typedesc)
                    target = content[0].astext()
                    xrefs = typed_field.make_xrefs(typed_field.typerolename, self.directive.domain, target, contnode=content[0], env=self.directive.state.document.settings.env)
                    if _is_single_paragraph(field_body):
                        paragraph = cast(nodes.paragraph, field_body[0])
                        paragraph.clear()
                        paragraph.extend(xrefs)
                    else:
                        field_body.clear()
                        field_body += nodes.paragraph('', '', *xrefs)
                continue
            typename = typedesc.name
            if is_typefield:
                content = [n for n in content if isinstance(n, (nodes.Inline, nodes.Text))]
                if content:
                    types.setdefault(typename, {})[fieldarg] = content
                continue
            if typedesc.is_typed:
                try:
                    argtype, argname = fieldarg.rsplit(None, 1)
                except ValueError:
                    pass
                else:
                    types.setdefault(typename, {})[argname] = [nodes.Text(argtype)]
                    fieldarg = argname
            translatable_content = nodes.inline(field_body.rawsource, translatable=True)
            translatable_content.document = field_body.parent.document
            translatable_content.source = field_body.parent.source
            translatable_content.line = field_body.parent.line
            translatable_content += content
            if typedesc.is_grouped:
                if typename in groupindices:
                    group = cast(Tuple[Field, List, Node], entries[groupindices[typename]])
                else:
                    groupindices[typename] = len(entries)
                    group = (typedesc, [], field)
                    entries.append(group)
                new_entry = typedesc.make_entry(fieldarg, [translatable_content])
                group[1].append(new_entry)
            else:
                new_entry = typedesc.make_entry(fieldarg, [translatable_content])
                entries.append((typedesc, new_entry, field))
        new_list = nodes.field_list()
        for entry in entries:
            if isinstance(entry, nodes.field):
                new_list += entry
            else:
                fieldtype, items, location = entry
                fieldtypes = types.get(fieldtype.name, {})
                env = self.directive.state.document.settings.env
                inliner = self.directive.state.inliner
                new_list += fieldtype.make_field(fieldtypes, self.directive.domain, items, env=env, inliner=inliner, location=location)
        node.replace_self(new_list)