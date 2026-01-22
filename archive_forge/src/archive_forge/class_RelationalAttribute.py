import re
import datetime
import numpy as np
import csv
import ctypes
class RelationalAttribute(Attribute):

    def __init__(self, name):
        super().__init__(name)
        self.type_name = 'relational'
        self.dtype = np.object_
        self.attributes = []
        self.dialect = None

    @classmethod
    def parse_attribute(cls, name, attr_string):
        """
        Parse the attribute line if it knows how. Returns the parsed
        attribute, or None.

        For date attributes, the attribute string would be like
        'date <format>'.
        """
        attr_string_lower = attr_string.lower().strip()
        if attr_string_lower[:len('relational')] == 'relational':
            return cls(name)
        else:
            return None

    def parse_data(self, data_str):
        elems = list(range(len(self.attributes)))
        escaped_string = data_str.encode().decode('unicode-escape')
        row_tuples = []
        for raw in escaped_string.split('\n'):
            row, self.dialect = split_data_line(raw, self.dialect)
            row_tuples.append(tuple([self.attributes[i].parse_data(row[i]) for i in elems]))
        return np.array(row_tuples, [(a.name, a.dtype) for a in self.attributes])

    def __str__(self):
        return super().__str__() + '\n\t' + '\n\t'.join((str(a) for a in self.attributes))