from plotly.basedatatypes import BaseTraceHierarchyType as _BaseTraceHierarchyType
import copy as _copy
@tickvals.setter
def tickvals(self, val):
    self['tickvals'] = val