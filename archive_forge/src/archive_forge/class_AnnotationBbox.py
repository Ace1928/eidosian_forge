import functools
import numpy as np
import matplotlib as mpl
from matplotlib import _api, _docstring
import matplotlib.artist as martist
import matplotlib.path as mpath
import matplotlib.text as mtext
import matplotlib.transforms as mtransforms
from matplotlib.font_manager import FontProperties
from matplotlib.image import BboxImage
from matplotlib.patches import (
from matplotlib.transforms import Bbox, BboxBase, TransformedBbox
class AnnotationBbox(martist.Artist, mtext._AnnotationBase):
    """
    Container for an `OffsetBox` referring to a specific position *xy*.

    Optionally an arrow pointing from the offsetbox to *xy* can be drawn.

    This is like `.Annotation`, but with `OffsetBox` instead of `.Text`.
    """
    zorder = 3

    def __str__(self):
        return f'AnnotationBbox({self.xy[0]:g},{self.xy[1]:g})'

    @_docstring.dedent_interpd
    def __init__(self, offsetbox, xy, xybox=None, xycoords='data', boxcoords=None, *, frameon=True, pad=0.4, annotation_clip=None, box_alignment=(0.5, 0.5), bboxprops=None, arrowprops=None, fontsize=None, **kwargs):
        """
        Parameters
        ----------
        offsetbox : `OffsetBox`

        xy : (float, float)
            The point *(x, y)* to annotate. The coordinate system is determined
            by *xycoords*.

        xybox : (float, float), default: *xy*
            The position *(x, y)* to place the text at. The coordinate system
            is determined by *boxcoords*.

        xycoords : single or two-tuple of str or `.Artist` or `.Transform` or callable, default: 'data'
            The coordinate system that *xy* is given in. See the parameter
            *xycoords* in `.Annotation` for a detailed description.

        boxcoords : single or two-tuple of str or `.Artist` or `.Transform` or callable, default: value of *xycoords*
            The coordinate system that *xybox* is given in. See the parameter
            *textcoords* in `.Annotation` for a detailed description.

        frameon : bool, default: True
            By default, the text is surrounded by a white `.FancyBboxPatch`
            (accessible as the ``patch`` attribute of the `.AnnotationBbox`).
            If *frameon* is set to False, this patch is made invisible.

        annotation_clip: bool or None, default: None
            Whether to clip (i.e. not draw) the annotation when the annotation
            point *xy* is outside the axes area.

            - If *True*, the annotation will be clipped when *xy* is outside
              the axes.
            - If *False*, the annotation will always be drawn.
            - If *None*, the annotation will be clipped when *xy* is outside
              the axes and *xycoords* is 'data'.

        pad : float, default: 0.4
            Padding around the offsetbox.

        box_alignment : (float, float)
            A tuple of two floats for a vertical and horizontal alignment of
            the offset box w.r.t. the *boxcoords*.
            The lower-left corner is (0, 0) and upper-right corner is (1, 1).

        bboxprops : dict, optional
            A dictionary of properties to set for the annotation bounding box,
            for example *boxstyle* and *alpha*.  See `.FancyBboxPatch` for
            details.

        arrowprops: dict, optional
            Arrow properties, see `.Annotation` for description.

        fontsize: float or str, optional
            Translated to points and passed as *mutation_scale* into
            `.FancyBboxPatch` to scale attributes of the box style (e.g. pad
            or rounding_size).  The name is chosen in analogy to `.Text` where
            *fontsize* defines the mutation scale as well.  If not given,
            :rc:`legend.fontsize` is used.  See `.Text.set_fontsize` for valid
            values.

        **kwargs
            Other `AnnotationBbox` properties.  See `.AnnotationBbox.set` for
            a list.
        """
        martist.Artist.__init__(self)
        mtext._AnnotationBase.__init__(self, xy, xycoords=xycoords, annotation_clip=annotation_clip)
        self.offsetbox = offsetbox
        self.arrowprops = arrowprops.copy() if arrowprops is not None else None
        self.set_fontsize(fontsize)
        self.xybox = xybox if xybox is not None else xy
        self.boxcoords = boxcoords if boxcoords is not None else xycoords
        self._box_alignment = box_alignment
        if arrowprops is not None:
            self._arrow_relpos = self.arrowprops.pop('relpos', (0.5, 0.5))
            self.arrow_patch = FancyArrowPatch((0, 0), (1, 1), **self.arrowprops)
        else:
            self._arrow_relpos = None
            self.arrow_patch = None
        self.patch = FancyBboxPatch(xy=(0.0, 0.0), width=1.0, height=1.0, facecolor='w', edgecolor='k', mutation_scale=self.prop.get_size_in_points(), snap=True, visible=frameon)
        self.patch.set_boxstyle('square', pad=pad)
        if bboxprops:
            self.patch.set(**bboxprops)
        self._internal_update(kwargs)

    @property
    def xyann(self):
        return self.xybox

    @xyann.setter
    def xyann(self, xyann):
        self.xybox = xyann
        self.stale = True

    @property
    def anncoords(self):
        return self.boxcoords

    @anncoords.setter
    def anncoords(self, coords):
        self.boxcoords = coords
        self.stale = True

    def contains(self, mouseevent):
        if self._different_canvas(mouseevent):
            return (False, {})
        if not self._check_xy(None):
            return (False, {})
        return self.offsetbox.contains(mouseevent)

    def get_children(self):
        children = [self.offsetbox, self.patch]
        if self.arrow_patch:
            children.append(self.arrow_patch)
        return children

    def set_figure(self, fig):
        if self.arrow_patch is not None:
            self.arrow_patch.set_figure(fig)
        self.offsetbox.set_figure(fig)
        martist.Artist.set_figure(self, fig)

    def set_fontsize(self, s=None):
        """
        Set the fontsize in points.

        If *s* is not given, reset to :rc:`legend.fontsize`.
        """
        if s is None:
            s = mpl.rcParams['legend.fontsize']
        self.prop = FontProperties(size=s)
        self.stale = True

    def get_fontsize(self):
        """Return the fontsize in points."""
        return self.prop.get_size_in_points()

    def get_window_extent(self, renderer=None):
        if renderer is None:
            renderer = self.figure._get_renderer()
        self.update_positions(renderer)
        return Bbox.union([child.get_window_extent(renderer) for child in self.get_children()])

    def get_tightbbox(self, renderer=None):
        if renderer is None:
            renderer = self.figure._get_renderer()
        self.update_positions(renderer)
        return Bbox.union([child.get_tightbbox(renderer) for child in self.get_children()])

    def update_positions(self, renderer):
        """Update pixel positions for the annotated point, the text, and the arrow."""
        ox0, oy0 = self._get_xy(renderer, self.xybox, self.boxcoords)
        bbox = self.offsetbox.get_bbox(renderer)
        fw, fh = self._box_alignment
        self.offsetbox.set_offset((ox0 - fw * bbox.width - bbox.x0, oy0 - fh * bbox.height - bbox.y0))
        bbox = self.offsetbox.get_window_extent(renderer)
        self.patch.set_bounds(bbox.bounds)
        mutation_scale = renderer.points_to_pixels(self.get_fontsize())
        self.patch.set_mutation_scale(mutation_scale)
        if self.arrowprops:
            arrow_begin = bbox.p0 + bbox.size * self._arrow_relpos
            arrow_end = self._get_position_xy(renderer)
            self.arrow_patch.set_positions(arrow_begin, arrow_end)
            if 'mutation_scale' in self.arrowprops:
                mutation_scale = renderer.points_to_pixels(self.arrowprops['mutation_scale'])
            self.arrow_patch.set_mutation_scale(mutation_scale)
            patchA = self.arrowprops.get('patchA', self.patch)
            self.arrow_patch.set_patchA(patchA)

    def draw(self, renderer):
        if renderer is not None:
            self._renderer = renderer
        if not self.get_visible() or not self._check_xy(renderer):
            return
        renderer.open_group(self.__class__.__name__, gid=self.get_gid())
        self.update_positions(renderer)
        if self.arrow_patch is not None:
            if self.arrow_patch.figure is None and self.figure is not None:
                self.arrow_patch.figure = self.figure
            self.arrow_patch.draw(renderer)
        self.patch.draw(renderer)
        self.offsetbox.draw(renderer)
        renderer.close_group(self.__class__.__name__)
        self.stale = False