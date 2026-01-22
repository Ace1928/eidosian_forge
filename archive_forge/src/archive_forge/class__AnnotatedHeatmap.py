import plotly.colors as clrs
from plotly import exceptions, optional_imports
from plotly.figure_factory import utils
from plotly.graph_objs import graph_objs
from plotly.validators.heatmap import ColorscaleValidator
class _AnnotatedHeatmap(object):
    """
    Refer to TraceFactory.create_annotated_heatmap() for docstring
    """

    def __init__(self, z, x, y, annotation_text, colorscale, font_colors, reversescale, **kwargs):
        self.z = z
        if x:
            self.x = x
        else:
            self.x = range(len(z[0]))
        if y:
            self.y = y
        else:
            self.y = range(len(z))
        if annotation_text is not None:
            self.annotation_text = annotation_text
        else:
            self.annotation_text = self.z
        self.colorscale = colorscale
        self.reversescale = reversescale
        self.font_colors = font_colors
        if np and isinstance(self.z, np.ndarray):
            self.zmin = np.amin(self.z)
            self.zmax = np.amax(self.z)
        else:
            self.zmin = min([v for row in self.z for v in row])
            self.zmax = max([v for row in self.z for v in row])
        if kwargs.get('zmin', None) is not None:
            self.zmin = kwargs['zmin']
        if kwargs.get('zmax', None) is not None:
            self.zmax = kwargs['zmax']
        self.zmid = (self.zmax + self.zmin) / 2
        if kwargs.get('zmid', None) is not None:
            self.zmid = kwargs['zmid']

    def get_text_color(self):
        """
        Get font color for annotations.

        The annotated heatmap can feature two text colors: min_text_color and
        max_text_color. The min_text_color is applied to annotations for
        heatmap values < (max_value - min_value)/2. The user can define these
        two colors. Otherwise the colors are defined logically as black or
        white depending on the heatmap's colorscale.

        :rtype (string, string) min_text_color, max_text_color: text
            color for annotations for heatmap values <
            (max_value - min_value)/2 and text color for annotations for
            heatmap values >= (max_value - min_value)/2
        """
        colorscales = ['Greys', 'Greens', 'Blues', 'YIGnBu', 'YIOrRd', 'RdBu', 'Picnic', 'Jet', 'Hot', 'Blackbody', 'Earth', 'Electric', 'Viridis', 'Cividis']
        colorscales_reverse = ['Reds']
        white = '#FFFFFF'
        black = '#000000'
        if self.font_colors:
            min_text_color = self.font_colors[0]
            max_text_color = self.font_colors[-1]
        elif self.colorscale in colorscales and self.reversescale:
            min_text_color = black
            max_text_color = white
        elif self.colorscale in colorscales:
            min_text_color = white
            max_text_color = black
        elif self.colorscale in colorscales_reverse and self.reversescale:
            min_text_color = white
            max_text_color = black
        elif self.colorscale in colorscales_reverse:
            min_text_color = black
            max_text_color = white
        elif isinstance(self.colorscale, list):
            min_col = to_rgb_color_list(self.colorscale[0][1], [255, 255, 255])
            max_col = to_rgb_color_list(self.colorscale[-1][1], [255, 255, 255])
            if self.reversescale:
                min_col, max_col = (max_col, min_col)
            if should_use_black_text(min_col):
                min_text_color = black
            else:
                min_text_color = white
            if should_use_black_text(max_col):
                max_text_color = black
            else:
                max_text_color = white
        else:
            min_text_color = black
            max_text_color = black
        return (min_text_color, max_text_color)

    def make_annotations(self):
        """
        Get annotations for each cell of the heatmap with graph_objs.Annotation

        :rtype (list[dict]) annotations: list of annotations for each cell of
            the heatmap
        """
        min_text_color, max_text_color = _AnnotatedHeatmap.get_text_color(self)
        annotations = []
        for n, row in enumerate(self.z):
            for m, val in enumerate(row):
                font_color = min_text_color if val < self.zmid else max_text_color
                annotations.append(graph_objs.layout.Annotation(text=str(self.annotation_text[n][m]), x=self.x[m], y=self.y[n], xref='x1', yref='y1', font=dict(color=font_color), showarrow=False))
        return annotations