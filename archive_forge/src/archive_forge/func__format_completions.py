import argparse
import inspect
import numbers
from collections import (
from typing import (
from .ansi import (
from .constants import (
from .argparse_custom import (
from .command_definition import (
from .exceptions import (
from .table_creator import (
def _format_completions(self, arg_state: _ArgumentState, completions: Union[List[str], List[CompletionItem]]) -> List[str]:
    """Format CompletionItems into hint table"""
    if len(completions) < 2 or not all((isinstance(c, CompletionItem) for c in completions)):
        return cast(List[str], completions)
    completion_items = cast(List[CompletionItem], completions)
    all_nums = all((isinstance(c.orig_value, numbers.Number) for c in completion_items))
    if not self._cmd2_app.matches_sorted:
        if all_nums:
            completion_items.sort(key=lambda c: c.orig_value)
        else:
            completion_items.sort(key=self._cmd2_app.default_sort_key)
        self._cmd2_app.matches_sorted = True
    if len(completions) <= self._cmd2_app.max_completion_items:
        four_spaces = 4 * ' '
        destination = arg_state.action.metavar if arg_state.action.metavar else arg_state.action.dest
        if isinstance(destination, tuple):
            tuple_index = min(len(destination) - 1, arg_state.count)
            destination = destination[tuple_index]
        desc_header = arg_state.action.get_descriptive_header()
        if desc_header is None:
            desc_header = DEFAULT_DESCRIPTIVE_HEADER
        desc_header = desc_header.replace('\t', four_spaces)
        token_width = style_aware_wcswidth(destination)
        desc_width = widest_line(desc_header)
        for item in completion_items:
            token_width = max(style_aware_wcswidth(item), token_width)
            item.description = item.description.replace('\t', four_spaces)
            desc_width = max(widest_line(item.description), desc_width)
        cols = list()
        dest_alignment = HorizontalAlignment.RIGHT if all_nums else HorizontalAlignment.LEFT
        cols.append(Column(destination.upper(), width=token_width, header_horiz_align=dest_alignment, data_horiz_align=dest_alignment))
        cols.append(Column(desc_header, width=desc_width))
        hint_table = SimpleTable(cols, divider_char=self._cmd2_app.ruler)
        table_data = [[item, item.description] for item in completion_items]
        self._cmd2_app.formatted_completions = hint_table.generate_table(table_data, row_spacing=0)
    return cast(List[str], completions)