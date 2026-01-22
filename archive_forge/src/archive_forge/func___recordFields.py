from struct import pack, unpack, calcsize, error, Struct
import os
import sys
import time
import array
import tempfile
import logging
import io
from datetime import date
import zipfile
def __recordFields(self, fields=None):
    """Returns the necessary info required to unpack a record's fields,
        restricted to a subset of fieldnames 'fields' if specified. 
        Returns a list of field info tuples, a name-index lookup dict, 
        and a Struct instance for unpacking these fields. Note that DeletionFlag
        is not a valid field. 
        """
    if fields is not None:
        fields = list(set(fields))
        fmt, fmtSize = self.__recordFmt(fields=fields)
        recStruct = Struct(fmt)
        for name in fields:
            if name not in self.__fieldLookup or name == 'DeletionFlag':
                raise ValueError('"{}" is not a valid field name'.format(name))
        fieldTuples = []
        for fieldinfo in self.fields[1:]:
            name = fieldinfo[0]
            if name in fields:
                fieldTuples.append(fieldinfo)
        recLookup = dict(((f[0], i) for i, f in enumerate(fieldTuples)))
    else:
        fieldTuples = self.fields[1:]
        recStruct = self.__fullRecStruct
        recLookup = self.__fullRecLookup
    return (fieldTuples, recLookup, recStruct)