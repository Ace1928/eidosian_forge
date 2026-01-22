from the canonical composite structure in that we don't really have
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.pdfbase.pdfmetrics import stringWidth
from reportlab.graphics.shapes import Drawing, String, Line, Rect, Wedge, ArcPath
from reportlab.graphics.widgetbase import Widget
from Bio.Graphics import _write
from Bio.Graphics.GenomeDiagram import _Colors
def _overdraw_subcomponents(self, cur_drawing):
    """Draw any annotated features on the chromosome segment (PRIVATE).

        Assumes _draw_segment already called to fill out the basic shape,
        and assmes that uses the same boundaries.
        """
    segment_y = self.end_y_position
    segment_width = (self.end_x_position - self.start_x_position) * self.chr_percent
    label_sep = (self.end_x_position - self.start_x_position) * self.label_sep_percent
    segment_height = self.start_y_position - self.end_y_position
    segment_x = self.start_x_position + 0.5 * (self.end_x_position - self.start_x_position - segment_width)
    left_labels = []
    right_labels = []
    for f in self.features:
        try:
            start = f.location.start
            end = f.location.end
            strand = f.location.strand
            try:
                color = _color_trans.translate(f.qualifiers['color'][0])
            except Exception:
                color = self.default_feature_color
            fill_color = color
            name = ''
            for qualifier in self.name_qualifiers:
                if qualifier in f.qualifiers:
                    name = f.qualifiers[qualifier][0]
                    break
        except AttributeError:
            start, end, strand, name, color = f[:5]
            color = _color_trans.translate(color)
            if len(f) > 5:
                fill_color = _color_trans.translate(f[5])
            else:
                fill_color = color
        assert 0 <= start <= end <= self.bp_length
        if strand == +1:
            x = segment_x + segment_width * 0.6
            w = segment_width * 0.4
        elif strand == -1:
            x = segment_x
            w = segment_width * 0.4
        else:
            x = segment_x
            w = segment_width
        local_scale = segment_height / self.bp_length
        fill_rectangle = Rect(x, segment_y + segment_height - local_scale * start, w, local_scale * (start - end))
        fill_rectangle.fillColor = fill_color
        fill_rectangle.strokeColor = color
        cur_drawing.add(fill_rectangle)
        if name:
            if fill_color == color:
                back_color = None
            else:
                back_color = fill_color
            value = (segment_y + segment_height - local_scale * start, color, back_color, name)
            if strand == -1:
                self._left_labels.append(value)
            else:
                self._right_labels.append(value)