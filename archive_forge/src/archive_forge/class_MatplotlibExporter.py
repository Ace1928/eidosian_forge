from .. import PlotItem
from .. import functions as fn
from ..Qt import QtCore, QtWidgets
from .Exporter import Exporter
class MatplotlibExporter(Exporter):
    Name = 'Matplotlib Window'
    windows = []

    def __init__(self, item):
        Exporter.__init__(self, item)

    def parameters(self):
        return None

    def cleanAxes(self, axl):
        if type(axl) is not list:
            axl = [axl]
        for ax in axl:
            if ax is None:
                continue
            for loc, spine in ax.spines.items():
                if loc in ['left', 'bottom']:
                    pass
                elif loc in ['right', 'top']:
                    spine.set_color('none')
                else:
                    raise ValueError('Unknown spine location: %s' % loc)
                ax.xaxis.set_ticks_position('bottom')

    def export(self, fileName=None):
        if not isinstance(self.item, PlotItem):
            raise Exception('MatplotlibExporter currently only works with PlotItem')
        mpw = MatplotlibWindow()
        MatplotlibExporter.windows.append(mpw)
        fig = mpw.getFigure()
        xax = self.item.getAxis('bottom')
        yax = self.item.getAxis('left')
        xlabel = xax.label.toPlainText()
        ylabel = yax.label.toPlainText()
        title = self.item.titleLabel.text
        xscale = yscale = 1.0
        if xax.autoSIPrefix:
            xscale = xax.autoSIPrefixScale
        if yax.autoSIPrefix:
            yscale = yax.autoSIPrefixScale
        ax = fig.add_subplot(111, title=title)
        ax.clear()
        self.cleanAxes(ax)
        for item in self.item.curves:
            x, y = item.getData()
            x = x * xscale
            y = y * yscale
            opts = item.opts
            pen = fn.mkPen(opts['pen'])
            if pen.style() == QtCore.Qt.PenStyle.NoPen:
                linestyle = ''
            else:
                linestyle = '-'
            color = pen.color().getRgbF()
            symbol = opts['symbol']
            symbol = _symbol_pg_to_mpl.get(symbol, '')
            symbolPen = fn.mkPen(opts['symbolPen'])
            symbolBrush = fn.mkBrush(opts['symbolBrush'])
            markeredgecolor = symbolPen.color().getRgbF()
            markerfacecolor = symbolBrush.color().getRgbF()
            markersize = opts['symbolSize']
            if opts['fillLevel'] is not None and opts['fillBrush'] is not None:
                fillBrush = fn.mkBrush(opts['fillBrush'])
                fillcolor = fillBrush.color().getRgbF()
                ax.fill_between(x=x, y1=y, y2=opts['fillLevel'], facecolor=fillcolor)
            ax.plot(x, y, marker=symbol, color=color, linewidth=pen.width(), linestyle=linestyle, markeredgecolor=markeredgecolor, markerfacecolor=markerfacecolor, markersize=markersize)
            xr, yr = self.item.viewRange()
            ax.set_xbound(xr[0] * xscale, xr[1] * xscale)
            ax.set_ybound(yr[0] * yscale, yr[1] * yscale)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        mpw.draw()