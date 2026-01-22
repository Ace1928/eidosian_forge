import sys
from abc import ABC
from typing import TYPE_CHECKING, Any, List, Optional, Tuple, Union
from ..generic import ArrayObject, DictionaryObject
from ..generic._base import (
from ..generic._fit import DEFAULT_FIT, Fit
from ..generic._rectangle import RectangleObject
from ..generic._utils import hex_to_rgb
from ._base import NO_FLAGS, AnnotationDictionary
class FreeText(MarkupAnnotation):
    """A FreeText annotation"""

    def __init__(self, *, text: str, rect: Union[RectangleObject, Tuple[float, float, float, float]], font: str='Helvetica', bold: bool=False, italic: bool=False, font_size: str='14pt', font_color: str='000000', border_color: Optional[str]='000000', background_color: Optional[str]='ffffff', **kwargs: Any):
        super().__init__(**kwargs)
        self[NameObject('/Subtype')] = NameObject('/FreeText')
        self[NameObject('/Rect')] = RectangleObject(rect)
        font_str = 'font: '
        if bold is True:
            font_str = f'{font_str}bold '
        if italic is True:
            font_str = f'{font_str}italic '
        font_str = f'{font_str}{font} {font_size}'
        font_str = f'{font_str};text-align:left;color:#{font_color}'
        default_appearance_string = ''
        if border_color:
            for st in hex_to_rgb(border_color):
                default_appearance_string = f'{default_appearance_string}{st} '
            default_appearance_string = f'{default_appearance_string}rg'
        self.update({NameObject('/Subtype'): NameObject('/FreeText'), NameObject('/Rect'): RectangleObject(rect), NameObject('/Contents'): TextStringObject(text), NameObject('/DS'): TextStringObject(font_str), NameObject('/DA'): TextStringObject(default_appearance_string)})
        if border_color is None:
            self[NameObject('/BS')] = DictionaryObject({NameObject('/W'): NumberObject(0)})
        if background_color is not None:
            self[NameObject('/C')] = ArrayObject([FloatObject(n) for n in hex_to_rgb(background_color)])