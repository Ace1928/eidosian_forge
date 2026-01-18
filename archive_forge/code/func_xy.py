from plotly.basedatatypes import BaseTraceType as _BaseTraceType
import copy as _copy
@xy.setter
def xy(self, val):
    self['xy'] = val