import re
import datetime
import numpy as np
import csv
import ctypes
class NumericAttribute(Attribute):

    def __init__(self, name):
        super().__init__(name)
        self.type_name = 'numeric'
        self.dtype = np.float64

    @classmethod
    def parse_attribute(cls, name, attr_string):
        """
        Parse the attribute line if it knows how. Returns the parsed
        attribute, or None.

        For numeric attributes, the attribute string would be like
        'numeric' or 'int' or 'real'.
        """
        attr_string = attr_string.lower().strip()
        if attr_string[:len('numeric')] == 'numeric' or attr_string[:len('int')] == 'int' or attr_string[:len('real')] == 'real':
            return cls(name)
        else:
            return None

    def parse_data(self, data_str):
        """
        Parse a value of this type.

        Parameters
        ----------
        data_str : str
           string to convert

        Returns
        -------
        f : float
           where float can be nan

        Examples
        --------
        >>> from scipy.io.arff._arffread import NumericAttribute
        >>> atr = NumericAttribute('atr')
        >>> atr.parse_data('1')
        1.0
        >>> atr.parse_data('1\\n')
        1.0
        >>> atr.parse_data('?\\n')
        nan
        """
        if '?' in data_str:
            return np.nan
        else:
            return float(data_str)

    def _basic_stats(self, data):
        nbfac = data.size * 1.0 / (data.size - 1)
        return (np.nanmin(data), np.nanmax(data), np.mean(data), np.std(data) * nbfac)