import sys
import time
from collections import OrderedDict
from datetime import datetime, timedelta, timezone
import numpy as np
from .AxisItem import AxisItem
class DateAxisItem(AxisItem):
    """
    **Bases:** :class:`AxisItem <pyqtgraph.AxisItem>`
    
    An AxisItem that displays dates from unix timestamps.

    The display format is adjusted automatically depending on the current time
    density (seconds/point) on the axis. For more details on changing this
    behaviour, see :func:`setZoomLevelForDensity() <pyqtgraph.DateAxisItem.setZoomLevelForDensity>`.
    
    Can be added to an existing plot e.g. via 
    :func:`setAxisItems({'bottom':axis}) <pyqtgraph.PlotItem.setAxisItems>`.

    """

    def __init__(self, orientation='bottom', utcOffset=None, **kwargs):
        """
        Create a new DateAxisItem.
        
        For `orientation` and `**kwargs`, see
        :func:`AxisItem.__init__ <pyqtgraph.AxisItem.__init__>`.
        
        """
        super(DateAxisItem, self).__init__(orientation, **kwargs)
        if utcOffset is None:
            utcOffset = getOffsetFromUtc()
        self.utcOffset = utcOffset
        self.zoomLevels = OrderedDict([(np.inf, YEAR_MONTH_ZOOM_LEVEL), (5 * 3600 * 24, MONTH_DAY_ZOOM_LEVEL), (6 * 3600, DAY_HOUR_ZOOM_LEVEL), (15 * 60, HOUR_MINUTE_ZOOM_LEVEL), (30, HMS_ZOOM_LEVEL), (1, MS_ZOOM_LEVEL)])
        self.autoSIPrefix = False

    def tickStrings(self, values, scale, spacing):
        tickSpecs = self.zoomLevel.tickSpecs
        tickSpec = next((s for s in tickSpecs if s.spacing == spacing), None)
        try:
            dates = [utcfromtimestamp(v - self.utcOffset) for v in values]
        except (OverflowError, ValueError, OSError):
            return ['%g' % ((v - self.utcOffset) // SEC_PER_YEAR + 1970) for v in values]
        formatStrings = []
        for x in dates:
            try:
                s = x.strftime(tickSpec.format)
                if '%f' in tickSpec.format:
                    s = s[:-3]
                elif '%Y' in tickSpec.format:
                    s = s.lstrip('0')
                formatStrings.append(s)
            except ValueError:
                formatStrings.append('')
        return formatStrings

    def tickValues(self, minVal, maxVal, size):
        density = (maxVal - minVal) / size
        self.setZoomLevelForDensity(density)
        values = self.zoomLevel.tickValues(minVal, maxVal, minSpc=self.minSpacing)
        return values

    def setZoomLevelForDensity(self, density):
        """
        Setting `zoomLevel` and `minSpacing` based on given density of seconds per pixel
        
        The display format is adjusted automatically depending on the current time
        density (seconds/point) on the axis. You can customize the behaviour by 
        overriding this function or setting a different set of zoom levels
        than the default one. The `zoomLevels` variable is a dictionary with the
        maximal distance of ticks in seconds which are allowed for each zoom level
        before the axis switches to the next coarser level. To customize the zoom level
        selection, override this function.
        """
        padding = 10
        if self.orientation in ['bottom', 'top']:

            def sizeOf(text):
                return self.fontMetrics.boundingRect(text).width() + padding
        else:

            def sizeOf(text):
                return self.fontMetrics.boundingRect(text).height() + padding
        self.zoomLevel = YEAR_MONTH_ZOOM_LEVEL
        for maximalSpacing, zoomLevel in self.zoomLevels.items():
            size = sizeOf(zoomLevel.exampleText)
            if maximalSpacing / size < density:
                break
            self.zoomLevel = zoomLevel
        self.zoomLevel.utcOffset = self.utcOffset
        size = sizeOf(self.zoomLevel.exampleText)
        self.minSpacing = density * size

    def linkToView(self, view):
        """Link this axis to a ViewBox, causing its displayed range to match the visible range of the view."""
        self._linkToView_internal(view)
        _min = MIN_REGULAR_TIMESTAMP
        _max = MAX_REGULAR_TIMESTAMP
        if self.orientation in ['right', 'left']:
            view.setLimits(yMin=_min, yMax=_max)
        else:
            view.setLimits(xMin=_min, xMax=_max)

    def generateDrawSpecs(self, p):
        if self.style['tickFont'] is not None:
            p.setFont(self.style['tickFont'])
        self.fontMetrics = p.fontMetrics()
        return super(DateAxisItem, self).generateDrawSpecs(p)