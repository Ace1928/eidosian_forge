import math
import os
import re
import textwrap
from itertools import chain, groupby
from typing import (TYPE_CHECKING, Any, Dict, Generator, Iterable, List, Optional, Set, Tuple,
from docutils import nodes, writers
from docutils.nodes import Element, Text
from docutils.utils import column_width
from sphinx import addnodes
from sphinx.locale import _, admonitionlabels
from sphinx.util.docutils import SphinxTranslator
class TextTranslator(SphinxTranslator):
    builder: 'TextBuilder' = None

    def __init__(self, document: nodes.document, builder: 'TextBuilder') -> None:
        super().__init__(document, builder)
        newlines = self.config.text_newlines
        if newlines == 'windows':
            self.nl = '\r\n'
        elif newlines == 'native':
            self.nl = os.linesep
        else:
            self.nl = '\n'
        self.sectionchars = self.config.text_sectionchars
        self.add_secnumbers = self.config.text_add_secnumbers
        self.secnumber_suffix = self.config.text_secnumber_suffix
        self.states: List[List[Tuple[int, Union[str, List[str]]]]] = [[]]
        self.stateindent = [0]
        self.list_counter: List[int] = []
        self.sectionlevel = 0
        self.lineblocklevel = 0
        self.table: Table = None

    def add_text(self, text: str) -> None:
        self.states[-1].append((-1, text))

    def new_state(self, indent: int=STDINDENT) -> None:
        self.states.append([])
        self.stateindent.append(indent)

    def end_state(self, wrap: bool=True, end: List[str]=[''], first: str=None) -> None:
        content = self.states.pop()
        maxindent = sum(self.stateindent)
        indent = self.stateindent.pop()
        result: List[Tuple[int, List[str]]] = []
        toformat: List[str] = []

        def do_format() -> None:
            if not toformat:
                return
            if wrap:
                res = my_wrap(''.join(toformat), width=MAXWIDTH - maxindent)
            else:
                res = ''.join(toformat).splitlines()
            if end:
                res += end
            result.append((indent, res))
        for itemindent, item in content:
            if itemindent == -1:
                toformat.append(item)
            else:
                do_format()
                result.append((indent + itemindent, item))
                toformat = []
        do_format()
        if first is not None and result:
            newindent = result[0][0] - indent
            if result[0][1] == ['']:
                result.insert(0, (newindent, [first]))
            else:
                text = first + result[0][1].pop(0)
                result.insert(0, (newindent, [text]))
        self.states[-1].extend(result)

    def visit_document(self, node: Element) -> None:
        self.new_state(0)

    def depart_document(self, node: Element) -> None:
        self.end_state()
        self.body = self.nl.join((line and ' ' * indent + line for indent, lines in self.states[0] for line in lines))

    def visit_section(self, node: Element) -> None:
        self._title_char = self.sectionchars[self.sectionlevel]
        self.sectionlevel += 1

    def depart_section(self, node: Element) -> None:
        self.sectionlevel -= 1

    def visit_topic(self, node: Element) -> None:
        self.new_state(0)

    def depart_topic(self, node: Element) -> None:
        self.end_state()
    visit_sidebar = visit_topic
    depart_sidebar = depart_topic

    def visit_rubric(self, node: Element) -> None:
        self.new_state(0)
        self.add_text('-[ ')

    def depart_rubric(self, node: Element) -> None:
        self.add_text(' ]-')
        self.end_state()

    def visit_compound(self, node: Element) -> None:
        pass

    def depart_compound(self, node: Element) -> None:
        pass

    def visit_glossary(self, node: Element) -> None:
        pass

    def depart_glossary(self, node: Element) -> None:
        pass

    def visit_title(self, node: Element) -> None:
        if isinstance(node.parent, nodes.Admonition):
            self.add_text(node.astext() + ': ')
            raise nodes.SkipNode
        self.new_state(0)

    def get_section_number_string(self, node: Element) -> str:
        if isinstance(node.parent, nodes.section):
            anchorname = '#' + node.parent['ids'][0]
            numbers = self.builder.secnumbers.get(anchorname)
            if numbers is None:
                numbers = self.builder.secnumbers.get('')
            if numbers is not None:
                return '.'.join(map(str, numbers)) + self.secnumber_suffix
        return ''

    def depart_title(self, node: Element) -> None:
        if isinstance(node.parent, nodes.section):
            char = self._title_char
        else:
            char = '^'
        text = ''
        text = ''.join((x[1] for x in self.states.pop() if x[0] == -1))
        if self.add_secnumbers:
            text = self.get_section_number_string(node) + text
        self.stateindent.pop()
        title = ['', text, '%s' % (char * column_width(text)), '']
        if len(self.states) == 2 and len(self.states[-1]) == 0:
            title.pop(0)
        self.states[-1].append((0, title))

    def visit_subtitle(self, node: Element) -> None:
        pass

    def depart_subtitle(self, node: Element) -> None:
        pass

    def visit_attribution(self, node: Element) -> None:
        self.add_text('-- ')

    def depart_attribution(self, node: Element) -> None:
        pass

    def visit_desc(self, node: Element) -> None:
        pass

    def depart_desc(self, node: Element) -> None:
        pass

    def visit_desc_signature(self, node: Element) -> None:
        self.new_state(0)

    def depart_desc_signature(self, node: Element) -> None:
        self.end_state(wrap=False, end=None)

    def visit_desc_signature_line(self, node: Element) -> None:
        pass

    def depart_desc_signature_line(self, node: Element) -> None:
        self.add_text('\n')

    def visit_desc_content(self, node: Element) -> None:
        self.new_state()
        self.add_text(self.nl)

    def depart_desc_content(self, node: Element) -> None:
        self.end_state()

    def visit_desc_inline(self, node: Element) -> None:
        pass

    def depart_desc_inline(self, node: Element) -> None:
        pass

    def visit_desc_name(self, node: Element) -> None:
        pass

    def depart_desc_name(self, node: Element) -> None:
        pass

    def visit_desc_addname(self, node: Element) -> None:
        pass

    def depart_desc_addname(self, node: Element) -> None:
        pass

    def visit_desc_type(self, node: Element) -> None:
        pass

    def depart_desc_type(self, node: Element) -> None:
        pass

    def visit_desc_returns(self, node: Element) -> None:
        self.add_text(' -> ')

    def depart_desc_returns(self, node: Element) -> None:
        pass

    def visit_desc_parameterlist(self, node: Element) -> None:
        self.add_text('(')
        self.first_param = 1

    def depart_desc_parameterlist(self, node: Element) -> None:
        self.add_text(')')

    def visit_desc_parameter(self, node: Element) -> None:
        if not self.first_param:
            self.add_text(', ')
        else:
            self.first_param = 0
        self.add_text(node.astext())
        raise nodes.SkipNode

    def visit_desc_optional(self, node: Element) -> None:
        self.add_text('[')

    def depart_desc_optional(self, node: Element) -> None:
        self.add_text(']')

    def visit_desc_annotation(self, node: Element) -> None:
        pass

    def depart_desc_annotation(self, node: Element) -> None:
        pass

    def visit_figure(self, node: Element) -> None:
        self.new_state()

    def depart_figure(self, node: Element) -> None:
        self.end_state()

    def visit_caption(self, node: Element) -> None:
        pass

    def depart_caption(self, node: Element) -> None:
        pass

    def visit_productionlist(self, node: Element) -> None:
        self.new_state()
        names = []
        productionlist = cast(Iterable[addnodes.production], node)
        for production in productionlist:
            names.append(production['tokenname'])
        maxlen = max((len(name) for name in names))
        lastname = None
        for production in productionlist:
            if production['tokenname']:
                self.add_text(production['tokenname'].ljust(maxlen) + ' ::=')
                lastname = production['tokenname']
            elif lastname is not None:
                self.add_text('%s    ' % (' ' * len(lastname)))
            self.add_text(production.astext() + self.nl)
        self.end_state(wrap=False)
        raise nodes.SkipNode

    def visit_footnote(self, node: Element) -> None:
        label = cast(nodes.label, node[0])
        self._footnote = label.astext().strip()
        self.new_state(len(self._footnote) + 3)

    def depart_footnote(self, node: Element) -> None:
        self.end_state(first='[%s] ' % self._footnote)

    def visit_citation(self, node: Element) -> None:
        if len(node) and isinstance(node[0], nodes.label):
            self._citlabel = node[0].astext()
        else:
            self._citlabel = ''
        self.new_state(len(self._citlabel) + 3)

    def depart_citation(self, node: Element) -> None:
        self.end_state(first='[%s] ' % self._citlabel)

    def visit_label(self, node: Element) -> None:
        raise nodes.SkipNode

    def visit_legend(self, node: Element) -> None:
        pass

    def depart_legend(self, node: Element) -> None:
        pass

    def visit_option_list(self, node: Element) -> None:
        pass

    def depart_option_list(self, node: Element) -> None:
        pass

    def visit_option_list_item(self, node: Element) -> None:
        self.new_state(0)

    def depart_option_list_item(self, node: Element) -> None:
        self.end_state()

    def visit_option_group(self, node: Element) -> None:
        self._firstoption = True

    def depart_option_group(self, node: Element) -> None:
        self.add_text('     ')

    def visit_option(self, node: Element) -> None:
        if self._firstoption:
            self._firstoption = False
        else:
            self.add_text(', ')

    def depart_option(self, node: Element) -> None:
        pass

    def visit_option_string(self, node: Element) -> None:
        pass

    def depart_option_string(self, node: Element) -> None:
        pass

    def visit_option_argument(self, node: Element) -> None:
        self.add_text(node['delimiter'])

    def depart_option_argument(self, node: Element) -> None:
        pass

    def visit_description(self, node: Element) -> None:
        pass

    def depart_description(self, node: Element) -> None:
        pass

    def visit_tabular_col_spec(self, node: Element) -> None:
        raise nodes.SkipNode

    def visit_colspec(self, node: Element) -> None:
        self.table.colwidth.append(node['colwidth'])
        raise nodes.SkipNode

    def visit_tgroup(self, node: Element) -> None:
        pass

    def depart_tgroup(self, node: Element) -> None:
        pass

    def visit_thead(self, node: Element) -> None:
        pass

    def depart_thead(self, node: Element) -> None:
        pass

    def visit_tbody(self, node: Element) -> None:
        self.table.set_separator()

    def depart_tbody(self, node: Element) -> None:
        pass

    def visit_row(self, node: Element) -> None:
        if self.table.lines:
            self.table.add_row()

    def depart_row(self, node: Element) -> None:
        pass

    def visit_entry(self, node: Element) -> None:
        self.entry = Cell(rowspan=node.get('morerows', 0) + 1, colspan=node.get('morecols', 0) + 1)
        self.new_state(0)

    def depart_entry(self, node: Element) -> None:
        text = self.nl.join((self.nl.join(x[1]) for x in self.states.pop()))
        self.stateindent.pop()
        self.entry.text = text
        self.table.add_cell(self.entry)
        self.entry = None

    def visit_table(self, node: Element) -> None:
        if self.table:
            raise NotImplementedError('Nested tables are not supported.')
        self.new_state(0)
        self.table = Table()

    def depart_table(self, node: Element) -> None:
        self.add_text(str(self.table))
        self.table = None
        self.end_state(wrap=False)

    def visit_acks(self, node: Element) -> None:
        bullet_list = cast(nodes.bullet_list, node[0])
        list_items = cast(Iterable[nodes.list_item], bullet_list)
        self.new_state(0)
        self.add_text(', '.join((n.astext() for n in list_items)) + '.')
        self.end_state()
        raise nodes.SkipNode

    def visit_image(self, node: Element) -> None:
        if 'alt' in node.attributes:
            self.add_text(_('[image: %s]') % node['alt'])
        self.add_text(_('[image]'))
        raise nodes.SkipNode

    def visit_transition(self, node: Element) -> None:
        indent = sum(self.stateindent)
        self.new_state(0)
        self.add_text('=' * (MAXWIDTH - indent))
        self.end_state()
        raise nodes.SkipNode

    def visit_bullet_list(self, node: Element) -> None:
        self.list_counter.append(-1)

    def depart_bullet_list(self, node: Element) -> None:
        self.list_counter.pop()

    def visit_enumerated_list(self, node: Element) -> None:
        self.list_counter.append(node.get('start', 1) - 1)

    def depart_enumerated_list(self, node: Element) -> None:
        self.list_counter.pop()

    def visit_definition_list(self, node: Element) -> None:
        self.list_counter.append(-2)

    def depart_definition_list(self, node: Element) -> None:
        self.list_counter.pop()

    def visit_list_item(self, node: Element) -> None:
        if self.list_counter[-1] == -1:
            self.new_state(2)
        elif self.list_counter[-1] == -2:
            pass
        else:
            self.list_counter[-1] += 1
            self.new_state(len(str(self.list_counter[-1])) + 2)

    def depart_list_item(self, node: Element) -> None:
        if self.list_counter[-1] == -1:
            self.end_state(first='* ')
        elif self.list_counter[-1] == -2:
            pass
        else:
            self.end_state(first='%s. ' % self.list_counter[-1])

    def visit_definition_list_item(self, node: Element) -> None:
        self._classifier_count_in_li = len(list(node.findall(nodes.classifier)))

    def depart_definition_list_item(self, node: Element) -> None:
        pass

    def visit_term(self, node: Element) -> None:
        self.new_state(0)

    def depart_term(self, node: Element) -> None:
        if not self._classifier_count_in_li:
            self.end_state(end=None)

    def visit_classifier(self, node: Element) -> None:
        self.add_text(' : ')

    def depart_classifier(self, node: Element) -> None:
        self._classifier_count_in_li -= 1
        if not self._classifier_count_in_li:
            self.end_state(end=None)

    def visit_definition(self, node: Element) -> None:
        self.new_state()

    def depart_definition(self, node: Element) -> None:
        self.end_state()

    def visit_field_list(self, node: Element) -> None:
        pass

    def depart_field_list(self, node: Element) -> None:
        pass

    def visit_field(self, node: Element) -> None:
        pass

    def depart_field(self, node: Element) -> None:
        pass

    def visit_field_name(self, node: Element) -> None:
        self.new_state(0)

    def depart_field_name(self, node: Element) -> None:
        self.add_text(':')
        self.end_state(end=None)

    def visit_field_body(self, node: Element) -> None:
        self.new_state()

    def depart_field_body(self, node: Element) -> None:
        self.end_state()

    def visit_centered(self, node: Element) -> None:
        pass

    def depart_centered(self, node: Element) -> None:
        pass

    def visit_hlist(self, node: Element) -> None:
        pass

    def depart_hlist(self, node: Element) -> None:
        pass

    def visit_hlistcol(self, node: Element) -> None:
        pass

    def depart_hlistcol(self, node: Element) -> None:
        pass

    def visit_admonition(self, node: Element) -> None:
        self.new_state(0)

    def depart_admonition(self, node: Element) -> None:
        self.end_state()

    def _visit_admonition(self, node: Element) -> None:
        self.new_state(2)

    def _depart_admonition(self, node: Element) -> None:
        label = admonitionlabels[node.tagname]
        indent = sum(self.stateindent) + len(label)
        if len(self.states[-1]) == 1 and self.states[-1][0][0] == 0 and (MAXWIDTH - indent >= sum((len(s) for s in self.states[-1][0][1]))):
            self.stateindent[-1] += len(label)
            self.end_state(first=label + ': ')
        else:
            self.states[-1].insert(0, (0, [self.nl]))
            self.end_state(first=label + ':')
    visit_attention = _visit_admonition
    depart_attention = _depart_admonition
    visit_caution = _visit_admonition
    depart_caution = _depart_admonition
    visit_danger = _visit_admonition
    depart_danger = _depart_admonition
    visit_error = _visit_admonition
    depart_error = _depart_admonition
    visit_hint = _visit_admonition
    depart_hint = _depart_admonition
    visit_important = _visit_admonition
    depart_important = _depart_admonition
    visit_note = _visit_admonition
    depart_note = _depart_admonition
    visit_tip = _visit_admonition
    depart_tip = _depart_admonition
    visit_warning = _visit_admonition
    depart_warning = _depart_admonition
    visit_seealso = _visit_admonition
    depart_seealso = _depart_admonition

    def visit_versionmodified(self, node: Element) -> None:
        self.new_state(0)

    def depart_versionmodified(self, node: Element) -> None:
        self.end_state()

    def visit_literal_block(self, node: Element) -> None:
        self.new_state()

    def depart_literal_block(self, node: Element) -> None:
        self.end_state(wrap=False)

    def visit_doctest_block(self, node: Element) -> None:
        self.new_state(0)

    def depart_doctest_block(self, node: Element) -> None:
        self.end_state(wrap=False)

    def visit_line_block(self, node: Element) -> None:
        self.new_state()
        self.lineblocklevel += 1

    def depart_line_block(self, node: Element) -> None:
        self.lineblocklevel -= 1
        self.end_state(wrap=False, end=None)
        if not self.lineblocklevel:
            self.add_text('\n')

    def visit_line(self, node: Element) -> None:
        pass

    def depart_line(self, node: Element) -> None:
        self.add_text('\n')

    def visit_block_quote(self, node: Element) -> None:
        self.new_state()

    def depart_block_quote(self, node: Element) -> None:
        self.end_state()

    def visit_compact_paragraph(self, node: Element) -> None:
        pass

    def depart_compact_paragraph(self, node: Element) -> None:
        pass

    def visit_paragraph(self, node: Element) -> None:
        if not isinstance(node.parent, nodes.Admonition) or isinstance(node.parent, addnodes.seealso):
            self.new_state(0)

    def depart_paragraph(self, node: Element) -> None:
        if not isinstance(node.parent, nodes.Admonition) or isinstance(node.parent, addnodes.seealso):
            self.end_state()

    def visit_target(self, node: Element) -> None:
        raise nodes.SkipNode

    def visit_index(self, node: Element) -> None:
        raise nodes.SkipNode

    def visit_toctree(self, node: Element) -> None:
        raise nodes.SkipNode

    def visit_substitution_definition(self, node: Element) -> None:
        raise nodes.SkipNode

    def visit_pending_xref(self, node: Element) -> None:
        pass

    def depart_pending_xref(self, node: Element) -> None:
        pass

    def visit_reference(self, node: Element) -> None:
        if self.add_secnumbers:
            numbers = node.get('secnumber')
            if numbers is not None:
                self.add_text('.'.join(map(str, numbers)) + self.secnumber_suffix)

    def depart_reference(self, node: Element) -> None:
        pass

    def visit_number_reference(self, node: Element) -> None:
        text = nodes.Text(node.get('title', '#'))
        self.visit_Text(text)
        raise nodes.SkipNode

    def visit_download_reference(self, node: Element) -> None:
        pass

    def depart_download_reference(self, node: Element) -> None:
        pass

    def visit_emphasis(self, node: Element) -> None:
        self.add_text('*')

    def depart_emphasis(self, node: Element) -> None:
        self.add_text('*')

    def visit_literal_emphasis(self, node: Element) -> None:
        self.add_text('*')

    def depart_literal_emphasis(self, node: Element) -> None:
        self.add_text('*')

    def visit_strong(self, node: Element) -> None:
        self.add_text('**')

    def depart_strong(self, node: Element) -> None:
        self.add_text('**')

    def visit_literal_strong(self, node: Element) -> None:
        self.add_text('**')

    def depart_literal_strong(self, node: Element) -> None:
        self.add_text('**')

    def visit_abbreviation(self, node: Element) -> None:
        self.add_text('')

    def depart_abbreviation(self, node: Element) -> None:
        if node.hasattr('explanation'):
            self.add_text(' (%s)' % node['explanation'])

    def visit_manpage(self, node: Element) -> None:
        return self.visit_literal_emphasis(node)

    def depart_manpage(self, node: Element) -> None:
        return self.depart_literal_emphasis(node)

    def visit_title_reference(self, node: Element) -> None:
        self.add_text('*')

    def depart_title_reference(self, node: Element) -> None:
        self.add_text('*')

    def visit_literal(self, node: Element) -> None:
        self.add_text('"')

    def depart_literal(self, node: Element) -> None:
        self.add_text('"')

    def visit_subscript(self, node: Element) -> None:
        self.add_text('_')

    def depart_subscript(self, node: Element) -> None:
        pass

    def visit_superscript(self, node: Element) -> None:
        self.add_text('^')

    def depart_superscript(self, node: Element) -> None:
        pass

    def visit_footnote_reference(self, node: Element) -> None:
        self.add_text('[%s]' % node.astext())
        raise nodes.SkipNode

    def visit_citation_reference(self, node: Element) -> None:
        self.add_text('[%s]' % node.astext())
        raise nodes.SkipNode

    def visit_Text(self, node: Text) -> None:
        self.add_text(node.astext())

    def depart_Text(self, node: Text) -> None:
        pass

    def visit_generated(self, node: Element) -> None:
        pass

    def depart_generated(self, node: Element) -> None:
        pass

    def visit_inline(self, node: Element) -> None:
        if 'xref' in node['classes'] or 'term' in node['classes']:
            self.add_text('*')

    def depart_inline(self, node: Element) -> None:
        if 'xref' in node['classes'] or 'term' in node['classes']:
            self.add_text('*')

    def visit_container(self, node: Element) -> None:
        pass

    def depart_container(self, node: Element) -> None:
        pass

    def visit_problematic(self, node: Element) -> None:
        self.add_text('>>')

    def depart_problematic(self, node: Element) -> None:
        self.add_text('<<')

    def visit_system_message(self, node: Element) -> None:
        self.new_state(0)
        self.add_text('<SYSTEM MESSAGE: %s>' % node.astext())
        self.end_state()
        raise nodes.SkipNode

    def visit_comment(self, node: Element) -> None:
        raise nodes.SkipNode

    def visit_meta(self, node: Element) -> None:
        raise nodes.SkipNode

    def visit_raw(self, node: Element) -> None:
        if 'text' in node.get('format', '').split():
            self.new_state(0)
            self.add_text(node.astext())
            self.end_state(wrap=False)
        raise nodes.SkipNode

    def visit_math(self, node: Element) -> None:
        pass

    def depart_math(self, node: Element) -> None:
        pass

    def visit_math_block(self, node: Element) -> None:
        self.new_state()

    def depart_math_block(self, node: Element) -> None:
        self.end_state()