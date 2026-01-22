import sys
import os.path
import csv
from docutils import io, nodes, statemachine, utils
from docutils.utils.error_reporting import SafeString
from docutils.utils import SystemMessagePropagation
from docutils.parsers.rst import Directive
from docutils.parsers.rst import directives
class ListTable(Table):
    """
    Implement tables whose data is encoded as a uniform two-level bullet list.
    For further ideas, see
    http://docutils.sf.net/docs/dev/rst/alternatives.html#list-driven-tables
    """
    option_spec = {'header-rows': directives.nonnegative_int, 'stub-columns': directives.nonnegative_int, 'width': directives.length_or_percentage_or_unitless, 'widths': directives.value_or(('auto',), directives.positive_int_list), 'class': directives.class_option, 'name': directives.unchanged, 'align': align}

    def run(self):
        if not self.content:
            error = self.state_machine.reporter.error('The "%s" directive is empty; content required.' % self.name, nodes.literal_block(self.block_text, self.block_text), line=self.lineno)
            return [error]
        title, messages = self.make_title()
        node = nodes.Element()
        self.state.nested_parse(self.content, self.content_offset, node)
        try:
            num_cols, col_widths = self.check_list_content(node)
            table_data = [[item.children for item in row_list[0]] for row_list in node[0]]
            header_rows = self.options.get('header-rows', 0)
            stub_columns = self.options.get('stub-columns', 0)
            self.check_table_dimensions(table_data, header_rows, stub_columns)
        except SystemMessagePropagation as detail:
            return [detail.args[0]]
        table_node = self.build_table_from_list(table_data, col_widths, header_rows, stub_columns)
        if 'align' in self.options:
            table_node['align'] = self.options.get('align')
        table_node['classes'] += self.options.get('class', [])
        self.set_table_width(table_node)
        self.add_name(table_node)
        if title:
            table_node.insert(0, title)
        return [table_node] + messages

    def check_list_content(self, node):
        if len(node) != 1 or not isinstance(node[0], nodes.bullet_list):
            error = self.state_machine.reporter.error('Error parsing content block for the "%s" directive: exactly one bullet list expected.' % self.name, nodes.literal_block(self.block_text, self.block_text), line=self.lineno)
            raise SystemMessagePropagation(error)
        list_node = node[0]
        for item_index in range(len(list_node)):
            item = list_node[item_index]
            if len(item) != 1 or not isinstance(item[0], nodes.bullet_list):
                error = self.state_machine.reporter.error('Error parsing content block for the "%s" directive: two-level bullet list expected, but row %s does not contain a second-level bullet list.' % (self.name, item_index + 1), nodes.literal_block(self.block_text, self.block_text), line=self.lineno)
                raise SystemMessagePropagation(error)
            elif item_index:
                if len(item[0]) != num_cols:
                    error = self.state_machine.reporter.error('Error parsing content block for the "%s" directive: uniform two-level bullet list expected, but row %s does not contain the same number of items as row 1 (%s vs %s).' % (self.name, item_index + 1, len(item[0]), num_cols), nodes.literal_block(self.block_text, self.block_text), line=self.lineno)
                    raise SystemMessagePropagation(error)
            else:
                num_cols = len(item[0])
        col_widths = self.get_column_widths(num_cols)
        return (num_cols, col_widths)

    def build_table_from_list(self, table_data, col_widths, header_rows, stub_columns):
        table = nodes.table()
        if self.widths == 'auto':
            table['classes'] += ['colwidths-auto']
        elif self.widths:
            table['classes'] += ['colwidths-given']
        tgroup = nodes.tgroup(cols=len(col_widths))
        table += tgroup
        for col_width in col_widths:
            colspec = nodes.colspec()
            if col_width is not None:
                colspec.attributes['colwidth'] = col_width
            if stub_columns:
                colspec.attributes['stub'] = 1
                stub_columns -= 1
            tgroup += colspec
        rows = []
        for row in table_data:
            row_node = nodes.row()
            for cell in row:
                entry = nodes.entry()
                entry += cell
                row_node += entry
            rows.append(row_node)
        if header_rows:
            thead = nodes.thead()
            thead.extend(rows[:header_rows])
            tgroup += thead
        tbody = nodes.tbody()
        tbody.extend(rows[header_rows:])
        tgroup += tbody
        return table