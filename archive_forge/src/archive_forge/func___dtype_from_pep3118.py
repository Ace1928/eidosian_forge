import ast
import re
import sys
import warnings
from ..exceptions import DTypePromotionError
from .multiarray import dtype, array, ndarray, promote_types
def __dtype_from_pep3118(stream, is_subdtype):
    field_spec = dict(names=[], formats=[], offsets=[], itemsize=0)
    offset = 0
    common_alignment = 1
    is_padding = False
    while stream:
        value = None
        if stream.consume('}'):
            break
        shape = None
        if stream.consume('('):
            shape = stream.consume_until(')')
            shape = tuple(map(int, shape.split(',')))
        if stream.next in ('@', '=', '<', '>', '^', '!'):
            byteorder = stream.advance(1)
            if byteorder == '!':
                byteorder = '>'
            stream.byteorder = byteorder
        if stream.byteorder in ('@', '^'):
            type_map = _pep3118_native_map
            type_map_chars = _pep3118_native_typechars
        else:
            type_map = _pep3118_standard_map
            type_map_chars = _pep3118_standard_typechars
        itemsize_str = stream.consume_until(lambda c: not c.isdigit())
        if itemsize_str:
            itemsize = int(itemsize_str)
        else:
            itemsize = 1
        is_padding = False
        if stream.consume('T{'):
            value, align = __dtype_from_pep3118(stream, is_subdtype=True)
        elif stream.next in type_map_chars:
            if stream.next == 'Z':
                typechar = stream.advance(2)
            else:
                typechar = stream.advance(1)
            is_padding = typechar == 'x'
            dtypechar = type_map[typechar]
            if dtypechar in 'USV':
                dtypechar += '%d' % itemsize
                itemsize = 1
            numpy_byteorder = {'@': '=', '^': '='}.get(stream.byteorder, stream.byteorder)
            value = dtype(numpy_byteorder + dtypechar)
            align = value.alignment
        elif stream.next in _pep3118_unsupported_map:
            desc = _pep3118_unsupported_map[stream.next]
            raise NotImplementedError('Unrepresentable PEP 3118 data type {!r} ({})'.format(stream.next, desc))
        else:
            raise ValueError('Unknown PEP 3118 data type specifier %r' % stream.s)
        extra_offset = 0
        if stream.byteorder == '@':
            start_padding = -offset % align
            intra_padding = -value.itemsize % align
            offset += start_padding
            if intra_padding != 0:
                if itemsize > 1 or (shape is not None and _prod(shape) > 1):
                    value = _add_trailing_padding(value, intra_padding)
                else:
                    extra_offset += intra_padding
            common_alignment = _lcm(align, common_alignment)
        if itemsize != 1:
            value = dtype((value, (itemsize,)))
        if shape is not None:
            value = dtype((value, shape))
        if stream.consume(':'):
            name = stream.consume_until(':')
        else:
            name = None
        if not (is_padding and name is None):
            if name is not None and name in field_spec['names']:
                raise RuntimeError(f"Duplicate field name '{name}' in PEP3118 format")
            field_spec['names'].append(name)
            field_spec['formats'].append(value)
            field_spec['offsets'].append(offset)
        offset += value.itemsize
        offset += extra_offset
        field_spec['itemsize'] = offset
    if stream.byteorder == '@':
        field_spec['itemsize'] += -offset % common_alignment
    if field_spec['names'] == [None] and field_spec['offsets'][0] == 0 and (field_spec['itemsize'] == field_spec['formats'][0].itemsize) and (not is_subdtype):
        ret = field_spec['formats'][0]
    else:
        _fix_names(field_spec)
        ret = dtype(field_spec)
    return (ret, common_alignment)