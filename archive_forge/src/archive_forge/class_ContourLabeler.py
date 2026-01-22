from contextlib import ExitStack
import functools
import math
from numbers import Integral
import numpy as np
from numpy import ma
import matplotlib as mpl
from matplotlib import _api, _docstring
from matplotlib.backend_bases import MouseButton
from matplotlib.lines import Line2D
from matplotlib.path import Path
from matplotlib.text import Text
import matplotlib.ticker as ticker
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.collections as mcoll
import matplotlib.font_manager as font_manager
import matplotlib.cbook as cbook
import matplotlib.patches as mpatches
import matplotlib.transforms as mtransforms
class ContourLabeler:
    """Mixin to provide labelling capability to `.ContourSet`."""

    def clabel(self, levels=None, *, fontsize=None, inline=True, inline_spacing=5, fmt=None, colors=None, use_clabeltext=False, manual=False, rightside_up=True, zorder=None):
        """
        Label a contour plot.

        Adds labels to line contours in this `.ContourSet` (which inherits from
        this mixin class).

        Parameters
        ----------
        levels : array-like, optional
            A list of level values, that should be labeled. The list must be
            a subset of ``cs.levels``. If not given, all levels are labeled.

        fontsize : str or float, default: :rc:`font.size`
            Size in points or relative size e.g., 'smaller', 'x-large'.
            See `.Text.set_size` for accepted string values.

        colors : color or colors or None, default: None
            The label colors:

            - If *None*, the color of each label matches the color of
              the corresponding contour.

            - If one string color, e.g., *colors* = 'r' or *colors* =
              'red', all labels will be plotted in this color.

            - If a tuple of colors (string, float, RGB, etc), different labels
              will be plotted in different colors in the order specified.

        inline : bool, default: True
            If ``True`` the underlying contour is removed where the label is
            placed.

        inline_spacing : float, default: 5
            Space in pixels to leave on each side of label when placing inline.

            This spacing will be exact for labels at locations where the
            contour is straight, less so for labels on curved contours.

        fmt : `.Formatter` or str or callable or dict, optional
            How the levels are formatted:

            - If a `.Formatter`, it is used to format all levels at once, using
              its `.Formatter.format_ticks` method.
            - If a str, it is interpreted as a %-style format string.
            - If a callable, it is called with one level at a time and should
              return the corresponding label.
            - If a dict, it should directly map levels to labels.

            The default is to use a standard `.ScalarFormatter`.

        manual : bool or iterable, default: False
            If ``True``, contour labels will be placed manually using
            mouse clicks. Click the first button near a contour to
            add a label, click the second button (or potentially both
            mouse buttons at once) to finish adding labels. The third
            button can be used to remove the last label added, but
            only if labels are not inline. Alternatively, the keyboard
            can be used to select label locations (enter to end label
            placement, delete or backspace act like the third mouse button,
            and any other key will select a label location).

            *manual* can also be an iterable object of (x, y) tuples.
            Contour labels will be created as if mouse is clicked at each
            (x, y) position.

        rightside_up : bool, default: True
            If ``True``, label rotations will always be plus
            or minus 90 degrees from level.

        use_clabeltext : bool, default: False
            If ``True``, use `.Text.set_transform_rotates_text` to ensure that
            label rotation is updated whenever the axes aspect changes.

        zorder : float or None, default: ``(2 + contour.get_zorder())``
            zorder of the contour labels.

        Returns
        -------
        labels
            A list of `.Text` instances for the labels.
        """
        if fmt is None:
            fmt = ticker.ScalarFormatter(useOffset=False)
            fmt.create_dummy_axis()
        self.labelFmt = fmt
        self._use_clabeltext = use_clabeltext
        self.labelManual = manual
        self.rightside_up = rightside_up
        self._clabel_zorder = 2 + self.get_zorder() if zorder is None else zorder
        if levels is None:
            levels = self.levels
            indices = list(range(len(self.cvalues)))
        else:
            levlabs = list(levels)
            indices, levels = ([], [])
            for i, lev in enumerate(self.levels):
                if lev in levlabs:
                    indices.append(i)
                    levels.append(lev)
            if len(levels) < len(levlabs):
                raise ValueError(f"Specified levels {levlabs} don't match available levels {self.levels}")
        self.labelLevelList = levels
        self.labelIndiceList = indices
        self._label_font_props = font_manager.FontProperties(size=fontsize)
        if colors is None:
            self.labelMappable = self
            self.labelCValueList = np.take(self.cvalues, self.labelIndiceList)
        else:
            cmap = mcolors.ListedColormap(colors, N=len(self.labelLevelList))
            self.labelCValueList = list(range(len(self.labelLevelList)))
            self.labelMappable = cm.ScalarMappable(cmap=cmap, norm=mcolors.NoNorm())
        self.labelXYs = []
        if np.iterable(manual):
            for x, y in manual:
                self.add_label_near(x, y, inline, inline_spacing)
        elif manual:
            print('Select label locations manually using first mouse button.')
            print('End manual selection with second mouse button.')
            if not inline:
                print('Remove last label by clicking third mouse button.')
            mpl._blocking_input.blocking_input_loop(self.axes.figure, ['button_press_event', 'key_press_event'], timeout=-1, handler=functools.partial(_contour_labeler_event_handler, self, inline, inline_spacing))
        else:
            self.labels(inline, inline_spacing)
        return cbook.silent_list('text.Text', self.labelTexts)

    @_api.deprecated('3.7', alternative='cs.labelTexts[0].get_font()')
    @property
    def labelFontProps(self):
        return self._label_font_props

    @_api.deprecated('3.7', alternative='[cs.labelTexts[0].get_font().get_size()] * len(cs.labelLevelList)')
    @property
    def labelFontSizeList(self):
        return [self._label_font_props.get_size()] * len(self.labelLevelList)

    @_api.deprecated('3.7', alternative='cs.labelTexts')
    @property
    def labelTextsList(self):
        return cbook.silent_list('text.Text', self.labelTexts)

    def print_label(self, linecontour, labelwidth):
        """Return whether a contour is long enough to hold a label."""
        return len(linecontour) > 10 * labelwidth or (len(linecontour) and (np.ptp(linecontour, axis=0) > 1.2 * labelwidth).any())

    def too_close(self, x, y, lw):
        """Return whether a label is already near this location."""
        thresh = (1.2 * lw) ** 2
        return any(((x - loc[0]) ** 2 + (y - loc[1]) ** 2 < thresh for loc in self.labelXYs))

    def _get_nth_label_width(self, nth):
        """Return the width of the *nth* label, in pixels."""
        fig = self.axes.figure
        renderer = fig._get_renderer()
        return Text(0, 0, self.get_text(self.labelLevelList[nth], self.labelFmt), figure=fig, fontproperties=self._label_font_props).get_window_extent(renderer).width

    @_api.deprecated('3.7', alternative='Artist.set')
    def set_label_props(self, label, text, color):
        """Set the label properties - color, fontsize, text."""
        label.set_text(text)
        label.set_color(color)
        label.set_fontproperties(self._label_font_props)
        label.set_clip_box(self.axes.bbox)

    def get_text(self, lev, fmt):
        """Get the text of the label."""
        if isinstance(lev, str):
            return lev
        elif isinstance(fmt, dict):
            return fmt.get(lev, '%1.3f')
        elif callable(getattr(fmt, 'format_ticks', None)):
            return fmt.format_ticks([*self.labelLevelList, lev])[-1]
        elif callable(fmt):
            return fmt(lev)
        else:
            return fmt % lev

    def locate_label(self, linecontour, labelwidth):
        """
        Find good place to draw a label (relatively flat part of the contour).
        """
        ctr_size = len(linecontour)
        n_blocks = int(np.ceil(ctr_size / labelwidth)) if labelwidth > 1 else 1
        block_size = ctr_size if n_blocks == 1 else int(labelwidth)
        xx = np.resize(linecontour[:, 0], (n_blocks, block_size))
        yy = np.resize(linecontour[:, 1], (n_blocks, block_size))
        yfirst = yy[:, :1]
        ylast = yy[:, -1:]
        xfirst = xx[:, :1]
        xlast = xx[:, -1:]
        s = (yfirst - yy) * (xlast - xfirst) - (xfirst - xx) * (ylast - yfirst)
        l = np.hypot(xlast - xfirst, ylast - yfirst)
        with np.errstate(divide='ignore', invalid='ignore'):
            distances = (abs(s) / l).sum(axis=-1)
        hbsize = block_size // 2
        adist = np.argsort(distances)
        for idx in np.append(adist, adist[0]):
            x, y = (xx[idx, hbsize], yy[idx, hbsize])
            if not self.too_close(x, y, labelwidth):
                break
        return (x, y, (idx * block_size + hbsize) % ctr_size)

    def _split_path_and_get_label_rotation(self, path, idx, screen_pos, lw, spacing=5):
        """
        Prepare for insertion of a label at index *idx* of *path*.

        Parameters
        ----------
        path : Path
            The path where the label will be inserted, in data space.
        idx : int
            The vertex index after which the label will be inserted.
        screen_pos : (float, float)
            The position where the label will be inserted, in screen space.
        lw : float
            The label width, in screen space.
        spacing : float
            Extra spacing around the label, in screen space.

        Returns
        -------
        path : Path
            The path, broken so that the label can be drawn over it.
        angle : float
            The rotation of the label.

        Notes
        -----
        Both tasks are done together to avoid calculating path lengths multiple times,
        which is relatively costly.

        The method used here involves computing the path length along the contour in
        pixel coordinates and then looking (label width / 2) away from central point to
        determine rotation and then to break contour if desired.  The extra spacing is
        taken into account when breaking the path, but not when computing the angle.
        """
        if hasattr(self, '_old_style_split_collections'):
            vis = False
            for coll in self._old_style_split_collections:
                vis |= coll.get_visible()
                coll.remove()
            self.set_visible(vis)
            del self._old_style_split_collections
        xys = path.vertices
        codes = path.codes
        pos = self.get_transform().inverted().transform(screen_pos)
        if not np.allclose(pos, xys[idx]):
            xys = np.insert(xys, idx, pos, axis=0)
            codes = np.insert(codes, idx, Path.LINETO)
        movetos = (codes == Path.MOVETO).nonzero()[0]
        start = movetos[movetos <= idx][-1]
        try:
            stop = movetos[movetos > idx][0]
        except IndexError:
            stop = len(codes)
        cc_xys = xys[start:stop]
        idx -= start
        is_closed_path = codes[stop - 1] == Path.CLOSEPOLY
        if is_closed_path:
            cc_xys = np.concatenate([cc_xys[idx:-1], cc_xys[:idx + 1]])
            idx = 0

        def interp_vec(x, xp, fp):
            return [np.interp(x, xp, col) for col in fp.T]
        screen_xys = self.get_transform().transform(cc_xys)
        path_cpls = np.insert(np.cumsum(np.hypot(*np.diff(screen_xys, axis=0).T)), 0, 0)
        path_cpls -= path_cpls[idx]
        target_cpls = np.array([-lw / 2, lw / 2])
        if is_closed_path:
            target_cpls[0] += path_cpls[-1] - path_cpls[0]
        (sx0, sx1), (sy0, sy1) = interp_vec(target_cpls, path_cpls, screen_xys)
        angle = np.rad2deg(np.arctan2(sy1 - sy0, sx1 - sx0))
        if self.rightside_up:
            angle = (angle + 90) % 180 - 90
        target_cpls += [-spacing, +spacing]
        i0, i1 = np.interp(target_cpls, path_cpls, range(len(path_cpls)), left=-1, right=-1)
        i0 = math.floor(i0)
        i1 = math.ceil(i1)
        (x0, x1), (y0, y1) = interp_vec(target_cpls, path_cpls, cc_xys)
        new_xy_blocks = []
        new_code_blocks = []
        if is_closed_path:
            if i0 != -1 and i1 != -1:
                points = cc_xys[i1:i0 + 1]
                new_xy_blocks.extend([[(x1, y1)], points, [(x0, y0)]])
                nlines = len(points) + 1
                new_code_blocks.extend([[Path.MOVETO], [Path.LINETO] * nlines])
        else:
            if i0 != -1:
                new_xy_blocks.extend([cc_xys[:i0 + 1], [(x0, y0)]])
                new_code_blocks.extend([[Path.MOVETO], [Path.LINETO] * (i0 + 1)])
            if i1 != -1:
                new_xy_blocks.extend([[(x1, y1)], cc_xys[i1:]])
                new_code_blocks.extend([[Path.MOVETO], [Path.LINETO] * (len(cc_xys) - i1)])
        xys = np.concatenate([xys[:start], *new_xy_blocks, xys[stop:]])
        codes = np.concatenate([codes[:start], *new_code_blocks, codes[stop:]])
        return (angle, Path(xys, codes))

    @_api.deprecated('3.8')
    def calc_label_rot_and_inline(self, slc, ind, lw, lc=None, spacing=5):
        """
        Calculate the appropriate label rotation given the linecontour
        coordinates in screen units, the index of the label location and the
        label width.

        If *lc* is not None or empty, also break contours and compute
        inlining.

        *spacing* is the empty space to leave around the label, in pixels.

        Both tasks are done together to avoid calculating path lengths
        multiple times, which is relatively costly.

        The method used here involves computing the path length along the
        contour in pixel coordinates and then looking approximately (label
        width / 2) away from central point to determine rotation and then to
        break contour if desired.
        """
        if lc is None:
            lc = []
        hlw = lw / 2.0
        closed = _is_closed_polygon(slc)
        if closed:
            slc = np.concatenate([slc[ind:-1], slc[:ind + 1]])
            if len(lc):
                lc = np.concatenate([lc[ind:-1], lc[:ind + 1]])
            ind = 0
        pl = np.zeros(slc.shape[0], dtype=float)
        dx = np.diff(slc, axis=0)
        pl[1:] = np.cumsum(np.hypot(dx[:, 0], dx[:, 1]))
        pl = pl - pl[ind]
        xi = np.array([-hlw, hlw])
        if closed:
            dp = np.array([pl[-1], 0])
        else:
            dp = np.zeros_like(xi)
        (dx,), (dy,) = (np.diff(np.interp(dp + xi, pl, slc_col)) for slc_col in slc.T)
        rotation = np.rad2deg(np.arctan2(dy, dx))
        if self.rightside_up:
            rotation = (rotation + 90) % 180 - 90
        nlc = []
        if len(lc):
            xi = dp + xi + np.array([-spacing, spacing])
            I = np.interp(xi, pl, np.arange(len(pl)), left=-1, right=-1)
            I = [np.floor(I[0]).astype(int), np.ceil(I[1]).astype(int)]
            if I[0] != -1:
                xy1 = [np.interp(xi[0], pl, lc_col) for lc_col in lc.T]
            if I[1] != -1:
                xy2 = [np.interp(xi[1], pl, lc_col) for lc_col in lc.T]
            if closed:
                if all((i != -1 for i in I)):
                    nlc.append(np.vstack([xy2, lc[I[1]:I[0] + 1], xy1]))
            else:
                if I[0] != -1:
                    nlc.append(np.vstack([lc[:I[0] + 1], xy1]))
                if I[1] != -1:
                    nlc.append(np.vstack([xy2, lc[I[1]:]]))
        return (rotation, nlc)

    def add_label(self, x, y, rotation, lev, cvalue):
        """Add contour label without `.Text.set_transform_rotates_text`."""
        data_x, data_y = self.axes.transData.inverted().transform((x, y))
        t = Text(data_x, data_y, text=self.get_text(lev, self.labelFmt), rotation=rotation, horizontalalignment='center', verticalalignment='center', zorder=self._clabel_zorder, color=self.labelMappable.to_rgba(cvalue, alpha=self.get_alpha()), fontproperties=self._label_font_props, clip_box=self.axes.bbox)
        self.labelTexts.append(t)
        self.labelCValues.append(cvalue)
        self.labelXYs.append((x, y))
        self.axes.add_artist(t)

    def add_label_clabeltext(self, x, y, rotation, lev, cvalue):
        """Add contour label with `.Text.set_transform_rotates_text`."""
        self.add_label(x, y, rotation, lev, cvalue)
        t = self.labelTexts[-1]
        data_rotation, = self.axes.transData.inverted().transform_angles([rotation], [[x, y]])
        t.set(rotation=data_rotation, transform_rotates_text=True)

    def add_label_near(self, x, y, inline=True, inline_spacing=5, transform=None):
        """
        Add a label near the point ``(x, y)``.

        Parameters
        ----------
        x, y : float
            The approximate location of the label.
        inline : bool, default: True
            If *True* remove the segment of the contour beneath the label.
        inline_spacing : int, default: 5
            Space in pixels to leave on each side of label when placing
            inline. This spacing will be exact for labels at locations where
            the contour is straight, less so for labels on curved contours.
        transform : `.Transform` or `False`, default: ``self.axes.transData``
            A transform applied to ``(x, y)`` before labeling.  The default
            causes ``(x, y)`` to be interpreted as data coordinates.  `False`
            is a synonym for `.IdentityTransform`; i.e. ``(x, y)`` should be
            interpreted as display coordinates.
        """
        if transform is None:
            transform = self.axes.transData
        if transform:
            x, y = transform.transform((x, y))
        idx_level_min, idx_vtx_min, proj = self._find_nearest_contour((x, y), self.labelIndiceList)
        path = self._paths[idx_level_min]
        level = self.labelIndiceList.index(idx_level_min)
        label_width = self._get_nth_label_width(level)
        rotation, path = self._split_path_and_get_label_rotation(path, idx_vtx_min, proj, label_width, inline_spacing)
        self.add_label(*proj, rotation, self.labelLevelList[idx_level_min], self.labelCValueList[idx_level_min])
        if inline:
            self._paths[idx_level_min] = path

    def pop_label(self, index=-1):
        """Defaults to removing last label, but any index can be supplied"""
        self.labelCValues.pop(index)
        t = self.labelTexts.pop(index)
        t.remove()

    def labels(self, inline, inline_spacing):
        if self._use_clabeltext:
            add_label = self.add_label_clabeltext
        else:
            add_label = self.add_label
        for idx, (icon, lev, cvalue) in enumerate(zip(self.labelIndiceList, self.labelLevelList, self.labelCValueList)):
            trans = self.get_transform()
            label_width = self._get_nth_label_width(idx)
            additions = []
            for subpath in self._paths[icon]._iter_connected_components():
                screen_xys = trans.transform(subpath.vertices)
                if self.print_label(screen_xys, label_width):
                    x, y, idx = self.locate_label(screen_xys, label_width)
                    rotation, path = self._split_path_and_get_label_rotation(subpath, idx, (x, y), label_width, inline_spacing)
                    add_label(x, y, rotation, lev, cvalue)
                    if inline:
                        additions.append(path)
                else:
                    additions.append(subpath)
            if inline:
                self._paths[icon] = Path.make_compound_path(*additions)

    def remove(self):
        super().remove()
        for text in self.labelTexts:
            text.remove()