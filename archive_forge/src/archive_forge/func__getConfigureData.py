import copy, functools
from ast import literal_eval
from reportlab.lib import colors
from reportlab.lib.validators import isNumber, isNumberOrNone, isColorOrNone, isString,\
from reportlab.lib.utils import isStr, yieldNoneSplits
from reportlab.graphics.widgets.markers import uSymbol2Symbol, isSymbol
from reportlab.lib.attrmap import AttrMap, AttrMapValue
from reportlab.pdfbase.pdfmetrics import stringWidth
from reportlab.graphics.widgetbase import TypedPropertyCollection, PropHolder, tpcGetItem
from reportlab.graphics.shapes import Line, Rect, Group, Drawing, PolyLine
from reportlab.graphics.charts.axes import XCategoryAxis, YValueAxis, YCategoryAxis, XValueAxis
from reportlab.graphics.charts.textlabels import BarChartLabel, NoneOrInstanceOfNA_Label
from reportlab.graphics.charts.areas import PlotArea
from reportlab.graphics.charts.legends import _objStr
from reportlab import cmp
def _getConfigureData(self):
    cAStyle = self.categoryAxis.style
    data = self.data
    cc = max(list(map(len, data)))
    _data = data
    if cAStyle not in ('parallel', 'parallel_3d'):
        data = []

        def _accumulate(*D):
            pdata = max((len(d) for d in D)) * [0]
            ndata = pdata[:]
            for d in D:
                for i, v in enumerate(d):
                    v = v or 0
                    if v <= -1e-06:
                        ndata[i] += v
                    else:
                        pdata[i] += v
            data.append(ndata)
            data.append(pdata)
        if cAStyle == 'stacked':
            _accumulate(*_data)
        else:
            self.getSeriesOrder()
            for b in self._seriesOrder:
                _accumulate(*(_data[j] for j in b))
    self._configureData = data