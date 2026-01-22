from __future__ import annotations
from collections import defaultdict
import datetime
import json
from typing import (
from pandas.io.excel._base import ExcelWriter
from pandas.io.excel._util import (
def _process_style(self, style: dict[str, Any] | None) -> str | None:
    """Convert a style dictionary to a OpenDocument style sheet

        Parameters
        ----------
        style : Dict
            Style dictionary

        Returns
        -------
        style_key : str
            Unique style key for later reference in sheet
        """
    from odf.style import ParagraphProperties, Style, TableCellProperties, TextProperties
    if style is None:
        return None
    style_key = json.dumps(style)
    if style_key in self._style_dict:
        return self._style_dict[style_key]
    name = f'pd{len(self._style_dict) + 1}'
    self._style_dict[style_key] = name
    odf_style = Style(name=name, family='table-cell')
    if 'font' in style:
        font = style['font']
        if font.get('bold', False):
            odf_style.addElement(TextProperties(fontweight='bold'))
    if 'borders' in style:
        borders = style['borders']
        for side, thickness in borders.items():
            thickness_translation = {'thin': '0.75pt solid #000000'}
            odf_style.addElement(TableCellProperties(attributes={f'border{side}': thickness_translation[thickness]}))
    if 'alignment' in style:
        alignment = style['alignment']
        horizontal = alignment.get('horizontal')
        if horizontal:
            odf_style.addElement(ParagraphProperties(textalign=horizontal))
        vertical = alignment.get('vertical')
        if vertical:
            odf_style.addElement(TableCellProperties(verticalalign=vertical))
    self.book.styles.addElement(odf_style)
    return name