import struct
import numpy as np
import tempfile
import zlib
import warnings
def _read_record(f):
    """Function to read in a full record"""
    record = {'rectype': _read_long(f)}
    nextrec = _read_uint32(f)
    nextrec += _read_uint32(f).astype(np.int64) * 2 ** 32
    _skip_bytes(f, 4)
    if record['rectype'] not in RECTYPE_DICT:
        raise Exception('Unknown RECTYPE: %i' % record['rectype'])
    record['rectype'] = RECTYPE_DICT[record['rectype']]
    if record['rectype'] in ['VARIABLE', 'HEAP_DATA']:
        if record['rectype'] == 'VARIABLE':
            record['varname'] = _read_string(f)
        else:
            record['heap_index'] = _read_long(f)
            _skip_bytes(f, 4)
        rectypedesc = _read_typedesc(f)
        if rectypedesc['typecode'] == 0:
            if nextrec == f.tell():
                record['data'] = None
            else:
                raise ValueError('Unexpected type code: 0')
        else:
            varstart = _read_long(f)
            if varstart != 7:
                raise Exception('VARSTART is not 7')
            if rectypedesc['structure']:
                record['data'] = _read_structure(f, rectypedesc['array_desc'], rectypedesc['struct_desc'])
            elif rectypedesc['array']:
                record['data'] = _read_array(f, rectypedesc['typecode'], rectypedesc['array_desc'])
            else:
                dtype = rectypedesc['typecode']
                record['data'] = _read_data(f, dtype)
    elif record['rectype'] == 'TIMESTAMP':
        _skip_bytes(f, 4 * 256)
        record['date'] = _read_string(f)
        record['user'] = _read_string(f)
        record['host'] = _read_string(f)
    elif record['rectype'] == 'VERSION':
        record['format'] = _read_long(f)
        record['arch'] = _read_string(f)
        record['os'] = _read_string(f)
        record['release'] = _read_string(f)
    elif record['rectype'] == 'IDENTIFICATON':
        record['author'] = _read_string(f)
        record['title'] = _read_string(f)
        record['idcode'] = _read_string(f)
    elif record['rectype'] == 'NOTICE':
        record['notice'] = _read_string(f)
    elif record['rectype'] == 'DESCRIPTION':
        record['description'] = _read_string_data(f)
    elif record['rectype'] == 'HEAP_HEADER':
        record['nvalues'] = _read_long(f)
        record['indices'] = [_read_long(f) for _ in range(record['nvalues'])]
    elif record['rectype'] == 'COMMONBLOCK':
        record['nvars'] = _read_long(f)
        record['name'] = _read_string(f)
        record['varnames'] = [_read_string(f) for _ in range(record['nvars'])]
    elif record['rectype'] == 'END_MARKER':
        record['end'] = True
    elif record['rectype'] == 'UNKNOWN':
        warnings.warn('Skipping UNKNOWN record', stacklevel=3)
    elif record['rectype'] == 'SYSTEM_VARIABLE':
        warnings.warn('Skipping SYSTEM_VARIABLE record', stacklevel=3)
    else:
        raise Exception(f"record['rectype']={record['rectype']} not implemented")
    f.seek(nextrec)
    return record