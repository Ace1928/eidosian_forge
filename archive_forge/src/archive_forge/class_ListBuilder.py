from __future__ import annotations
import re
from typing import List, TYPE_CHECKING, Optional, Any
import pyglet
import pyglet.text.layout
from pyglet.gl import GL_TEXTURE0, glActiveTexture, glBindTexture, glEnable, GL_BLEND, glBlendFunc, GL_SRC_ALPHA, \
class ListBuilder:

    def begin(self, decoder, style):
        """Begin a list.

        :Parameters:
            `decoder` : `StructuredTextDecoder`
                Decoder.
            `style` : dict
                Style dictionary that applies over the entire list.

        """
        left_margin = decoder.current_style.get('margin_left') or 0
        tab_stops = decoder.current_style.get('tab_stops')
        if tab_stops:
            tab_stops = list(tab_stops)
        else:
            tab_stops = []
        tab_stops.append(left_margin + 50)
        style['margin_left'] = left_margin + 50
        style['indent'] = -30
        style['tab_stops'] = tab_stops

    def item(self, decoder, style, value=None):
        """Begin a list item.

        :Parameters:
            `decoder` : `StructuredTextDecoder`
                Decoder.
            `style` : dict
                Style dictionary that applies over the list item.
            `value` : str
                Optional value of the list item.  The meaning is list-type
                dependent.

        """
        mark = self.get_mark(value)
        if mark:
            decoder.add_text(mark)
        decoder.add_text('\t')

    def get_mark(self, value=None):
        """Get the mark text for the next list item.

        :Parameters:
            `value` : str
                Optional value of the list item.  The meaning is list-type
                dependent.

        :rtype: str
        """
        return ''