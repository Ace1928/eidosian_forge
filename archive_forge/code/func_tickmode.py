from plotly.basedatatypes import BaseTraceHierarchyType as _BaseTraceHierarchyType
import copy as _copy
@tickmode.setter
def tickmode(self, val):
    self['tickmode'] = val