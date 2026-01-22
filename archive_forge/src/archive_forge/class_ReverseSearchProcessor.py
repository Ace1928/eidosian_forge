from a buffer before the BufferControl will render it to the screen.
from __future__ import annotations
import re
from abc import ABCMeta, abstractmethod
from typing import TYPE_CHECKING, Callable, Hashable, cast
from prompt_toolkit.application.current import get_app
from prompt_toolkit.cache import SimpleCache
from prompt_toolkit.document import Document
from prompt_toolkit.filters import FilterOrBool, to_filter, vi_insert_multiple_mode
from prompt_toolkit.formatted_text import (
from prompt_toolkit.formatted_text.utils import fragment_list_len, fragment_list_to_text
from prompt_toolkit.search import SearchDirection
from prompt_toolkit.utils import to_int, to_str
from .utils import explode_text_fragments
class ReverseSearchProcessor(Processor):
    """
    Process to display the "(reverse-i-search)`...`:..." stuff around
    the search buffer.

    Note: This processor is meant to be applied to the BufferControl that
    contains the search buffer, it's not meant for the original input.
    """
    _excluded_input_processors: list[type[Processor]] = [HighlightSearchProcessor, HighlightSelectionProcessor, BeforeInput, AfterInput]

    def _get_main_buffer(self, buffer_control: BufferControl) -> BufferControl | None:
        from prompt_toolkit.layout.controls import BufferControl
        prev_control = get_app().layout.search_target_buffer_control
        if isinstance(prev_control, BufferControl) and prev_control.search_buffer_control == buffer_control:
            return prev_control
        return None

    def _content(self, main_control: BufferControl, ti: TransformationInput) -> UIContent:
        from prompt_toolkit.layout.controls import BufferControl
        excluded_processors = tuple(self._excluded_input_processors)

        def filter_processor(item: Processor) -> Processor | None:
            """Filter processors from the main control that we want to disable
            here. This returns either an accepted processor or None."""
            if isinstance(item, _MergedProcessor):
                accepted_processors = [filter_processor(p) for p in item.processors]
                return merge_processors([p for p in accepted_processors if p is not None])
            elif isinstance(item, ConditionalProcessor):
                p = filter_processor(item.processor)
                if p:
                    return ConditionalProcessor(p, item.filter)
            elif not isinstance(item, excluded_processors):
                return item
            return None
        filtered_processor = filter_processor(merge_processors(main_control.input_processors or []))
        highlight_processor = HighlightIncrementalSearchProcessor()
        if filtered_processor:
            new_processors = [filtered_processor, highlight_processor]
        else:
            new_processors = [highlight_processor]
        from .controls import SearchBufferControl
        assert isinstance(ti.buffer_control, SearchBufferControl)
        buffer_control = BufferControl(buffer=main_control.buffer, input_processors=new_processors, include_default_input_processors=False, lexer=main_control.lexer, preview_search=True, search_buffer_control=ti.buffer_control)
        return buffer_control.create_content(ti.width, ti.height, preview_search=True)

    def apply_transformation(self, ti: TransformationInput) -> Transformation:
        from .controls import SearchBufferControl
        assert isinstance(ti.buffer_control, SearchBufferControl), '`ReverseSearchProcessor` should be applied to a `SearchBufferControl` only.'
        source_to_display: SourceToDisplay | None
        display_to_source: DisplayToSource | None
        main_control = self._get_main_buffer(ti.buffer_control)
        if ti.lineno == 0 and main_control:
            content = self._content(main_control, ti)
            line_fragments = content.get_line(content.cursor_position.y)
            if main_control.search_state.direction == SearchDirection.FORWARD:
                direction_text = 'i-search'
            else:
                direction_text = 'reverse-i-search'
            fragments_before: StyleAndTextTuples = [('class:prompt.search', '('), ('class:prompt.search', direction_text), ('class:prompt.search', ')`')]
            fragments = fragments_before + [('class:prompt.search.text', fragment_list_to_text(ti.fragments)), ('', "': ")] + line_fragments
            shift_position = fragment_list_len(fragments_before)
            source_to_display = lambda i: i + shift_position
            display_to_source = lambda i: i - shift_position
        else:
            source_to_display = None
            display_to_source = None
            fragments = ti.fragments
        return Transformation(fragments, source_to_display=source_to_display, display_to_source=display_to_source)