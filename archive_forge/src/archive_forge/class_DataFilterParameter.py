from collections import OrderedDict
import numpy as np
from .. import functions as fn
from .. import parametertree as ptree
from ..Qt import QtCore
class DataFilterParameter(ptree.types.GroupParameter):
    """A parameter group that specifies a set of filters to apply to tabular data.
    """
    sigFilterChanged = QtCore.Signal(object)

    def __init__(self):
        self.fields = {}
        ptree.types.GroupParameter.__init__(self, name='Data Filter', addText='Add filter..', addList=[])
        self.sigTreeStateChanged.connect(self.filterChanged)

    def filterChanged(self):
        self.sigFilterChanged.emit(self)

    def addNew(self, name):
        mode = self.fields[name].get('mode', 'range')
        if mode == 'range':
            child = self.addChild(RangeFilterItem(name, self.fields[name]))
        elif mode == 'enum':
            child = self.addChild(EnumFilterItem(name, self.fields[name]))
        else:
            raise ValueError("field mode must be 'range' or 'enum'")
        return child

    def fieldNames(self):
        return self.fields.keys()

    def setFields(self, fields):
        """Set the list of fields that are available to be filtered.

        *fields* must be a dict or list of tuples that maps field names
        to a specification describing the field. Each specification is
        itself a dict with either ``'mode':'range'`` or ``'mode':'enum'``::

            filter.setFields([
                ('field1', {'mode': 'range'}),
                ('field2', {'mode': 'enum', 'values': ['val1', 'val2', 'val3']}),
                ('field3', {'mode': 'enum', 'values': {'val1':True, 'val2':False, 'val3':True}}),
            ])
        """
        with fn.SignalBlock(self.sigTreeStateChanged, self.filterChanged):
            self.fields = OrderedDict(fields)
            names = self.fieldNames()
            self.setAddList(names)
            for ch in self.children():
                name = ch.fieldName
                if name in fields:
                    ch.updateFilter(fields[name])
        self.sigFilterChanged.emit(self)

    def filterData(self, data):
        if len(data) == 0:
            return data
        return data[self.generateMask(data)]

    def generateMask(self, data):
        """Return a boolean mask indicating whether each item in *data* passes
        the filter critera.
        """
        mask = np.ones(len(data), dtype=bool)
        if len(data) == 0:
            return mask
        for fp in self:
            if fp.value() is False:
                continue
            mask &= fp.generateMask(data, mask.copy())
        return mask

    def describe(self):
        """Return a list of strings describing the currently enabled filters."""
        desc = []
        for fp in self:
            if fp.value() is False:
                continue
            desc.append(fp.describe())
        return desc