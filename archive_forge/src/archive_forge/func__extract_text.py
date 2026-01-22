import math
import re
import sys
from decimal import Decimal
from pathlib import Path
from typing import (
from ._cmap import build_char_map, unknown_char_map
from ._protocols import PdfCommonDocProtocol
from ._text_extraction import (
from ._utils import (
from .constants import AnnotationDictionaryAttributes as ADA
from .constants import ImageAttributes as IA
from .constants import PageAttributes as PG
from .constants import Resources as RES
from .errors import PageSizeNotDefinedError, PdfReadError
from .filters import _xobj_to_image
from .generic import (
def _extract_text(self, obj: Any, pdf: Any, orientations: Tuple[int, ...]=(0, 90, 180, 270), space_width: float=200.0, content_key: Optional[str]=PG.CONTENTS, visitor_operand_before: Optional[Callable[[Any, Any, Any, Any], None]]=None, visitor_operand_after: Optional[Callable[[Any, Any, Any, Any], None]]=None, visitor_text: Optional[Callable[[Any, Any, Any, Any, Any], None]]=None) -> str:
    """
        See extract_text for most arguments.

        Args:
            content_key: indicate the default key where to extract data
                None = the object; this allow to reuse the function on XObject
                default = "/Content"
        """
    text: str = ''
    output: str = ''
    rtl_dir: bool = False
    cmaps: Dict[str, Tuple[str, float, Union[str, Dict[int, str]], Dict[str, str], DictionaryObject]] = {}
    try:
        objr = obj
        while NameObject(PG.RESOURCES) not in objr:
            objr = objr['/Parent'].get_object()
        resources_dict = cast(DictionaryObject, objr[PG.RESOURCES])
    except Exception:
        return ''
    if '/Font' in resources_dict:
        for f in cast(DictionaryObject, resources_dict['/Font']):
            cmaps[f] = build_char_map(f, space_width, obj)
    cmap: Tuple[Union[str, Dict[int, str]], Dict[str, str], str, Optional[DictionaryObject]] = ('charmap', {}, 'NotInitialized', None)
    try:
        content = obj[content_key].get_object() if isinstance(content_key, str) else obj
        if not isinstance(content, ContentStream):
            content = ContentStream(content, pdf, 'bytes')
    except KeyError:
        return ''
    cm_matrix: List[float] = [1.0, 0.0, 0.0, 1.0, 0.0, 0.0]
    cm_stack = []
    tm_matrix: List[float] = [1.0, 0.0, 0.0, 1.0, 0.0, 0.0]
    cm_prev: List[float] = [1.0, 0.0, 0.0, 1.0, 0.0, 0.0]
    tm_prev: List[float] = [1.0, 0.0, 0.0, 1.0, 0.0, 0.0]
    memo_cm: List[float] = [1.0, 0.0, 0.0, 1.0, 0.0, 0.0]
    memo_tm: List[float] = [1.0, 0.0, 0.0, 1.0, 0.0, 0.0]
    char_scale = 1.0
    space_scale = 1.0
    _space_width: float = 500.0
    TL = 0.0
    font_size = 12.0

    def current_spacewidth() -> float:
        return _space_width / 1000.0

    def process_operation(operator: bytes, operands: List[Any]) -> None:
        nonlocal cm_matrix, cm_stack, tm_matrix, cm_prev, tm_prev, memo_cm, memo_tm
        nonlocal char_scale, space_scale, _space_width, TL, font_size, cmap
        nonlocal orientations, rtl_dir, visitor_text, output, text
        global CUSTOM_RTL_MIN, CUSTOM_RTL_MAX, CUSTOM_RTL_SPECIAL_CHARS
        check_crlf_space: bool = False
        if operator == b'BT':
            tm_matrix = [1.0, 0.0, 0.0, 1.0, 0.0, 0.0]
            output += text
            if visitor_text is not None:
                visitor_text(text, memo_cm, memo_tm, cmap[3], font_size)
            text = ''
            memo_cm = cm_matrix.copy()
            memo_tm = tm_matrix.copy()
            return None
        elif operator == b'ET':
            output += text
            if visitor_text is not None:
                visitor_text(text, memo_cm, memo_tm, cmap[3], font_size)
            text = ''
            memo_cm = cm_matrix.copy()
            memo_tm = tm_matrix.copy()
        elif operator == b'q':
            cm_stack.append((cm_matrix, cmap, font_size, char_scale, space_scale, _space_width, TL))
        elif operator == b'Q':
            try:
                cm_matrix, cmap, font_size, char_scale, space_scale, _space_width, TL = cm_stack.pop()
            except Exception:
                cm_matrix = [1.0, 0.0, 0.0, 1.0, 0.0, 0.0]
        elif operator == b'cm':
            output += text
            if visitor_text is not None:
                visitor_text(text, memo_cm, memo_tm, cmap[3], font_size)
            text = ''
            cm_matrix = mult([float(operands[0]), float(operands[1]), float(operands[2]), float(operands[3]), float(operands[4]), float(operands[5])], cm_matrix)
            memo_cm = cm_matrix.copy()
            memo_tm = tm_matrix.copy()
        elif operator == b'Tz':
            char_scale = float(operands[0]) / 100.0
        elif operator == b'Tw':
            space_scale = 1.0 + float(operands[0])
        elif operator == b'TL':
            TL = float(operands[0])
        elif operator == b'Tf':
            if text != '':
                output += text
                if visitor_text is not None:
                    visitor_text(text, memo_cm, memo_tm, cmap[3], font_size)
            text = ''
            memo_cm = cm_matrix.copy()
            memo_tm = tm_matrix.copy()
            try:
                charMapTuple = cmaps[operands[0]]
                _space_width = charMapTuple[1]
                cmap = (charMapTuple[2], charMapTuple[3], operands[0], charMapTuple[4])
            except KeyError:
                _space_width = unknown_char_map[1]
                cmap = (unknown_char_map[2], unknown_char_map[3], '???' + operands[0], None)
            try:
                font_size = float(operands[1])
            except Exception:
                pass
        elif operator == b'Td':
            check_crlf_space = True
            tx = float(operands[0])
            ty = float(operands[1])
            tm_matrix[4] += tx * tm_matrix[0] + ty * tm_matrix[2]
            tm_matrix[5] += tx * tm_matrix[1] + ty * tm_matrix[3]
        elif operator == b'Tm':
            check_crlf_space = True
            tm_matrix = [float(operands[0]), float(operands[1]), float(operands[2]), float(operands[3]), float(operands[4]), float(operands[5])]
        elif operator == b'T*':
            check_crlf_space = True
            tm_matrix[5] -= TL
        elif operator == b'Tj':
            check_crlf_space = True
            text, rtl_dir = handle_tj(text, operands, cm_matrix, tm_matrix, cmap, orientations, output, font_size, rtl_dir, visitor_text)
        else:
            return None
        if check_crlf_space:
            try:
                text, output, cm_prev, tm_prev = crlf_space_check(text, (cm_prev, tm_prev), (cm_matrix, tm_matrix), (memo_cm, memo_tm), cmap, orientations, output, font_size, visitor_text, current_spacewidth())
                if text == '':
                    memo_cm = cm_matrix.copy()
                    memo_tm = tm_matrix.copy()
            except OrientationNotFoundError:
                return None
    for operands, operator in content.operations:
        if visitor_operand_before is not None:
            visitor_operand_before(operator, operands, cm_matrix, tm_matrix)
        if operator == b"'":
            process_operation(b'T*', [])
            process_operation(b'Tj', operands)
        elif operator == b'"':
            process_operation(b'Tw', [operands[0]])
            process_operation(b'Tc', [operands[1]])
            process_operation(b'T*', [])
            process_operation(b'Tj', operands[2:])
        elif operator == b'TD':
            process_operation(b'TL', [-operands[1]])
            process_operation(b'Td', operands)
        elif operator == b'TJ':
            for op in operands[0]:
                if isinstance(op, (str, bytes)):
                    process_operation(b'Tj', [op])
                if isinstance(op, (int, float, NumberObject, FloatObject)) and (abs(float(op)) >= _space_width and len(text) > 0 and (text[-1] != ' ')):
                    process_operation(b'Tj', [' '])
        elif operator == b'Do':
            output += text
            if visitor_text is not None:
                visitor_text(text, memo_cm, memo_tm, cmap[3], font_size)
            try:
                if output[-1] != '\n':
                    output += '\n'
                    if visitor_text is not None:
                        visitor_text('\n', memo_cm, memo_tm, cmap[3], font_size)
            except IndexError:
                pass
            try:
                xobj = resources_dict['/XObject']
                if xobj[operands[0]]['/Subtype'] != '/Image':
                    text = self.extract_xform_text(xobj[operands[0]], orientations, space_width, visitor_operand_before, visitor_operand_after, visitor_text)
                    output += text
                    if visitor_text is not None:
                        visitor_text(text, memo_cm, memo_tm, cmap[3], font_size)
            except Exception:
                logger_warning(f' impossible to decode XFormObject {operands[0]}', __name__)
            finally:
                text = ''
                memo_cm = cm_matrix.copy()
                memo_tm = tm_matrix.copy()
        else:
            process_operation(operator, operands)
        if visitor_operand_after is not None:
            visitor_operand_after(operator, operands, cm_matrix, tm_matrix)
    output += text
    if text != '' and visitor_text is not None:
        visitor_text(text, memo_cm, memo_tm, cmap[3], font_size)
    return output