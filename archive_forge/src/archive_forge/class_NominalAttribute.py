import re
import datetime
import numpy as np
import csv
import ctypes
class NominalAttribute(Attribute):
    type_name = 'nominal'

    def __init__(self, name, values):
        super().__init__(name)
        self.values = values
        self.range = values
        self.dtype = (np.bytes_, max((len(i) for i in values)))

    @staticmethod
    def _get_nom_val(atrv):
        """Given a string containing a nominal type, returns a tuple of the
        possible values.

        A nominal type is defined as something framed between braces ({}).

        Parameters
        ----------
        atrv : str
           Nominal type definition

        Returns
        -------
        poss_vals : tuple
           possible values

        Examples
        --------
        >>> from scipy.io.arff._arffread import NominalAttribute
        >>> NominalAttribute._get_nom_val("{floup, bouga, fl, ratata}")
        ('floup', 'bouga', 'fl', 'ratata')
        """
        m = r_nominal.match(atrv)
        if m:
            attrs, _ = split_data_line(m.group(1))
            return tuple(attrs)
        else:
            raise ValueError('This does not look like a nominal string')

    @classmethod
    def parse_attribute(cls, name, attr_string):
        """
        Parse the attribute line if it knows how. Returns the parsed
        attribute, or None.

        For nominal attributes, the attribute string would be like '{<attr_1>,
         <attr2>, <attr_3>}'.
        """
        if attr_string[0] == '{':
            values = cls._get_nom_val(attr_string)
            return cls(name, values)
        else:
            return None

    def parse_data(self, data_str):
        """
        Parse a value of this type.
        """
        if data_str in self.values:
            return data_str
        elif data_str == '?':
            return data_str
        else:
            raise ValueError(f'{str(data_str)} value not in {str(self.values)}')

    def __str__(self):
        msg = self.name + ',{'
        for i in range(len(self.values) - 1):
            msg += self.values[i] + ','
        msg += self.values[-1]
        msg += '}'
        return msg