from plotly import exceptions, optional_imports
from plotly.figure_factory import utils
from plotly.graph_objs import graph_objs
class _Distplot(object):
    """
    Refer to TraceFactory.create_distplot() for docstring
    """

    def __init__(self, hist_data, histnorm, group_labels, bin_size, curve_type, colors, rug_text, show_hist, show_curve):
        self.hist_data = hist_data
        self.histnorm = histnorm
        self.group_labels = group_labels
        self.bin_size = bin_size
        self.show_hist = show_hist
        self.show_curve = show_curve
        self.trace_number = len(hist_data)
        if rug_text:
            self.rug_text = rug_text
        else:
            self.rug_text = [None] * self.trace_number
        self.start = []
        self.end = []
        if colors:
            self.colors = colors
        else:
            self.colors = ['rgb(31, 119, 180)', 'rgb(255, 127, 14)', 'rgb(44, 160, 44)', 'rgb(214, 39, 40)', 'rgb(148, 103, 189)', 'rgb(140, 86, 75)', 'rgb(227, 119, 194)', 'rgb(127, 127, 127)', 'rgb(188, 189, 34)', 'rgb(23, 190, 207)']
        self.curve_x = [None] * self.trace_number
        self.curve_y = [None] * self.trace_number
        for trace in self.hist_data:
            self.start.append(min(trace) * 1.0)
            self.end.append(max(trace) * 1.0)

    def make_hist(self):
        """
        Makes the histogram(s) for FigureFactory.create_distplot().

        :rtype (list) hist: list of histogram representations
        """
        hist = [None] * self.trace_number
        for index in range(self.trace_number):
            hist[index] = dict(type='histogram', x=self.hist_data[index], xaxis='x1', yaxis='y1', histnorm=self.histnorm, name=self.group_labels[index], legendgroup=self.group_labels[index], marker=dict(color=self.colors[index % len(self.colors)]), autobinx=False, xbins=dict(start=self.start[index], end=self.end[index], size=self.bin_size[index]), opacity=0.7)
        return hist

    def make_kde(self):
        """
        Makes the kernel density estimation(s) for create_distplot().

        This is called when curve_type = 'kde' in create_distplot().

        :rtype (list) curve: list of kde representations
        """
        curve = [None] * self.trace_number
        for index in range(self.trace_number):
            self.curve_x[index] = [self.start[index] + x * (self.end[index] - self.start[index]) / 500 for x in range(500)]
            self.curve_y[index] = scipy_stats.gaussian_kde(self.hist_data[index])(self.curve_x[index])
            if self.histnorm == ALTERNATIVE_HISTNORM:
                self.curve_y[index] *= self.bin_size[index]
        for index in range(self.trace_number):
            curve[index] = dict(type='scatter', x=self.curve_x[index], y=self.curve_y[index], xaxis='x1', yaxis='y1', mode='lines', name=self.group_labels[index], legendgroup=self.group_labels[index], showlegend=False if self.show_hist else True, marker=dict(color=self.colors[index % len(self.colors)]))
        return curve

    def make_normal(self):
        """
        Makes the normal curve(s) for create_distplot().

        This is called when curve_type = 'normal' in create_distplot().

        :rtype (list) curve: list of normal curve representations
        """
        curve = [None] * self.trace_number
        mean = [None] * self.trace_number
        sd = [None] * self.trace_number
        for index in range(self.trace_number):
            mean[index], sd[index] = scipy_stats.norm.fit(self.hist_data[index])
            self.curve_x[index] = [self.start[index] + x * (self.end[index] - self.start[index]) / 500 for x in range(500)]
            self.curve_y[index] = scipy_stats.norm.pdf(self.curve_x[index], loc=mean[index], scale=sd[index])
            if self.histnorm == ALTERNATIVE_HISTNORM:
                self.curve_y[index] *= self.bin_size[index]
        for index in range(self.trace_number):
            curve[index] = dict(type='scatter', x=self.curve_x[index], y=self.curve_y[index], xaxis='x1', yaxis='y1', mode='lines', name=self.group_labels[index], legendgroup=self.group_labels[index], showlegend=False if self.show_hist else True, marker=dict(color=self.colors[index % len(self.colors)]))
        return curve

    def make_rug(self):
        """
        Makes the rug plot(s) for create_distplot().

        :rtype (list) rug: list of rug plot representations
        """
        rug = [None] * self.trace_number
        for index in range(self.trace_number):
            rug[index] = dict(type='scatter', x=self.hist_data[index], y=[self.group_labels[index]] * len(self.hist_data[index]), xaxis='x1', yaxis='y2', mode='markers', name=self.group_labels[index], legendgroup=self.group_labels[index], showlegend=False if self.show_hist or self.show_curve else True, text=self.rug_text[index], marker=dict(color=self.colors[index % len(self.colors)], symbol='line-ns-open'))
        return rug