from matplotlib.backend_bases import RendererBase
from matplotlib import colors as mcolors
from matplotlib import patches as mpatches
from matplotlib import transforms as mtransforms
from matplotlib.path import Path
import numpy as np
class PathEffectRenderer(RendererBase):
    """
    Implements a Renderer which contains another renderer.

    This proxy then intercepts draw calls, calling the appropriate
    :class:`AbstractPathEffect` draw method.

    .. note::
        Not all methods have been overridden on this RendererBase subclass.
        It may be necessary to add further methods to extend the PathEffects
        capabilities further.
    """

    def __init__(self, path_effects, renderer):
        """
        Parameters
        ----------
        path_effects : iterable of :class:`AbstractPathEffect`
            The path effects which this renderer represents.
        renderer : `~matplotlib.backend_bases.RendererBase` subclass

        """
        self._path_effects = path_effects
        self._renderer = renderer

    def copy_with_path_effect(self, path_effects):
        return self.__class__(path_effects, self._renderer)

    def draw_path(self, gc, tpath, affine, rgbFace=None):
        for path_effect in self._path_effects:
            path_effect.draw_path(self._renderer, gc, tpath, affine, rgbFace)

    def draw_markers(self, gc, marker_path, marker_trans, path, *args, **kwargs):
        if len(self._path_effects) == 1:
            return super().draw_markers(gc, marker_path, marker_trans, path, *args, **kwargs)
        for path_effect in self._path_effects:
            renderer = self.copy_with_path_effect([path_effect])
            renderer.draw_markers(gc, marker_path, marker_trans, path, *args, **kwargs)

    def draw_path_collection(self, gc, master_transform, paths, *args, **kwargs):
        if len(self._path_effects) == 1:
            return super().draw_path_collection(gc, master_transform, paths, *args, **kwargs)
        for path_effect in self._path_effects:
            renderer = self.copy_with_path_effect([path_effect])
            renderer.draw_path_collection(gc, master_transform, paths, *args, **kwargs)

    def _draw_text_as_path(self, gc, x, y, s, prop, angle, ismath):
        path, transform = self._get_text_path_transform(x, y, s, prop, angle, ismath)
        color = gc.get_rgb()
        gc.set_linewidth(0.0)
        self.draw_path(gc, path, transform, rgbFace=color)

    def __getattribute__(self, name):
        if name in ['flipy', 'get_canvas_width_height', 'new_gc', 'points_to_pixels', '_text2path', 'height', 'width']:
            return getattr(self._renderer, name)
        else:
            return object.__getattribute__(self, name)