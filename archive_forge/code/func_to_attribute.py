import re
import datetime
import numpy as np
import csv
import ctypes
def to_attribute(name, attr_string):
    attr_classes = (NominalAttribute, NumericAttribute, DateAttribute, StringAttribute, RelationalAttribute)
    for cls in attr_classes:
        attr = cls.parse_attribute(name, attr_string)
        if attr is not None:
            return attr
    raise ParseArffError('unknown attribute %s' % attr_string)