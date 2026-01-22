import os
import mmap
import struct
import codecs
class TableClass:
    size = struct.calcsize(fmt)

    def __init__(self, data, offset):
        items = struct.unpack(fmt, data[offset:offset + self.size])
        self.pairs = list(zip(names, items))
        for pname, pvalue in self.pairs:
            if isinstance(pvalue, bytes):
                pvalue = pvalue.decode('utf-8')
            setattr(self, pname, pvalue)

    def __repr__(self):
        return '{' + ', '.join([f'{pname} = {pvalue}' for pname, pvalue in self.pairs]) + '}'

    @staticmethod
    def array(data, offset, count):
        tables = []
        for i in range(count):
            tables.append(TableClass(data, offset))
            offset += TableClass.size
        return tables