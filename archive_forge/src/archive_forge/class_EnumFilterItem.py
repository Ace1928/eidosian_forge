from collections import OrderedDict
import numpy as np
from .. import functions as fn
from .. import parametertree as ptree
from ..Qt import QtCore
class EnumFilterItem(ptree.types.SimpleParameter):

    def __init__(self, name, opts):
        self.fieldName = name
        ptree.types.SimpleParameter.__init__(self, name=name, autoIncrementName=True, type='bool', value=True, removable=True, renamable=True)
        self.setEnumVals(opts)

    def generateMask(self, data, startMask):
        vals = data[self.fieldName][startMask]
        mask = np.ones(len(vals), dtype=bool)
        otherMask = np.ones(len(vals), dtype=bool)
        for c in self:
            key = c.maskValue
            if key == '__other__':
                m = ~otherMask
            else:
                m = vals != key
                otherMask &= m
            if c.value() is False:
                mask &= m
        startMask[startMask] = mask
        return startMask

    def describe(self):
        vals = [ch.name() for ch in self if ch.value() is True]
        return '%s: %s' % (self.fieldName, ', '.join(vals))

    def updateFilter(self, opts):
        self.setEnumVals(opts)

    def setEnumVals(self, opts):
        vals = opts.get('values', {})
        prevState = {}
        for ch in self.children():
            prevState[ch.name()] = ch.value()
            self.removeChild(ch)
        if not isinstance(vals, dict):
            vals = OrderedDict([(v, (str(v), True)) for v in vals])
        for val, valopts in vals.items():
            if isinstance(valopts, bool):
                enabled = valopts
                vname = str(val)
            elif isinstance(valopts, str):
                enabled = True
                vname = valopts
            elif isinstance(valopts, tuple):
                vname, enabled = valopts
            ch = ptree.Parameter.create(name=vname, type='bool', value=prevState.get(vname, enabled))
            ch.maskValue = val
            self.addChild(ch)
        ch = ptree.Parameter.create(name='(other)', type='bool', value=prevState.get('(other)', True))
        ch.maskValue = '__other__'
        self.addChild(ch)