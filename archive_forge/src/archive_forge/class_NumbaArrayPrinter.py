import re
class NumbaArrayPrinter:

    def __init__(self, val):
        self.val = val

    def to_string(self):
        try:
            import numpy as np
            HAVE_NUMPY = True
        except ImportError:
            HAVE_NUMPY = False
        try:
            NULL = 0
            data = self.val['data']
            itemsize = self.val['itemsize']
            rshp = self.val['shape']
            rstrides = self.val['strides']
            is_aligned = False
            ty_str = str(self.val.type)
            if HAVE_NUMPY and ('aligned' in ty_str or 'Record' in ty_str):
                ty_str = ty_str.replace('unaligned ', '').strip()
                matcher = re.compile('array\\((Record.*), (.*), (.*)\\)\\ \\(.*')
                arr_info = [x.strip() for x in matcher.match(ty_str).groups()]
                dtype_str, ndim_str, order_str = arr_info
                rstr = 'Record\\((.*\\[.*\\]);([0-9]+);(True|False)'
                rstr_match = re.match(rstr, dtype_str)
                fields, balign, is_aligned_str = rstr_match.groups()
                is_aligned = is_aligned_str == 'True'
                field_dts = fields.split(',')
                struct_entries = []
                for f in field_dts:
                    splitted = f.split('[')
                    name = splitted[0]
                    dt_part = splitted[1:]
                    if len(dt_part) > 1:
                        raise TypeError('Unsupported sub-type: %s' % f)
                    else:
                        dt_part = dt_part[0]
                        if 'nestedarray' in dt_part:
                            raise TypeError('Unsupported sub-type: %s' % f)
                        dt_as_str = dt_part.split(';')[0].split('=')[1]
                        dtype = np.dtype(dt_as_str)
                    struct_entries.append((name, dtype))
                dtype_str = struct_entries
            else:
                matcher = re.compile('array\\((.*),(.*),(.*)\\)\\ \\(.*')
                arr_info = [x.strip() for x in matcher.match(ty_str).groups()]
                dtype_str, ndim_str, order_str = arr_info
                if 'unichr x ' in dtype_str:
                    dtype_str = dtype_str[1:-1].replace('unichr x ', '<U')

            def dwarr2inttuple(dwarr):
                fields = dwarr.type.fields()
                lo, hi = fields[0].type.range()
                return tuple([int(dwarr[x]) for x in range(lo, hi + 1)])
            shape = dwarr2inttuple(rshp)
            strides = dwarr2inttuple(rstrides)
            if data != NULL:
                if HAVE_NUMPY:
                    shp_arr = np.array([max(0, x - 1) for x in shape])
                    strd_arr = np.array(strides)
                    extent = np.sum(shp_arr * strd_arr)
                    extent += int(itemsize)
                    dtype_clazz = np.dtype(dtype_str, align=is_aligned)
                    dtype = dtype_clazz
                    this_proc = gdb.selected_inferior()
                    mem = this_proc.read_memory(int(data), extent)
                    arr_data = np.frombuffer(mem, dtype=dtype)
                    new_arr = np.lib.stride_tricks.as_strided(arr_data, shape=shape, strides=strides)
                    return '\n' + str(new_arr)
                return 'array([...], dtype=%s, shape=%s)' % (dtype_str, shape)
            else:
                buf = list(['NULL/Uninitialized'])
                return 'array([' + ', '.join(buf) + ']' + ')'
        except Exception as e:
            return 'array[Exception: Failed to parse. %s]' % e