import numpy as np
def _struct_dict_str(dtype, includealignedflag):
    names = dtype.names
    fld_dtypes = []
    offsets = []
    titles = []
    for name in names:
        fld_dtype, offset, title = _unpack_field(*dtype.fields[name])
        fld_dtypes.append(fld_dtype)
        offsets.append(offset)
        titles.append(title)
    if np.core.arrayprint._get_legacy_print_mode() <= 121:
        colon = ':'
        fieldsep = ','
    else:
        colon = ': '
        fieldsep = ', '
    ret = "{'names'%s[" % colon
    ret += fieldsep.join((repr(name) for name in names))
    ret += "], 'formats'%s[" % colon
    ret += fieldsep.join((_construction_repr(fld_dtype, short=True) for fld_dtype in fld_dtypes))
    ret += "], 'offsets'%s[" % colon
    ret += fieldsep.join(('%d' % offset for offset in offsets))
    if any((title is not None for title in titles)):
        ret += "], 'titles'%s[" % colon
        ret += fieldsep.join((repr(title) for title in titles))
    ret += "], 'itemsize'%s%d" % (colon, dtype.itemsize)
    if includealignedflag and dtype.isalignedstruct:
        ret += ", 'aligned'%sTrue}" % colon
    else:
        ret += '}'
    return ret