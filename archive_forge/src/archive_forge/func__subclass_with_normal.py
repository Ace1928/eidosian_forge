from matplotlib.backend_bases import RendererBase
from matplotlib import colors as mcolors
from matplotlib import patches as mpatches
from matplotlib import transforms as mtransforms
from matplotlib.path import Path
import numpy as np
def _subclass_with_normal(effect_class):
    """
    Create a PathEffect class combining *effect_class* and a normal draw.
    """

    class withEffect(effect_class):

        def draw_path(self, renderer, gc, tpath, affine, rgbFace):
            super().draw_path(renderer, gc, tpath, affine, rgbFace)
            renderer.draw_path(gc, tpath, affine, rgbFace)
    withEffect.__name__ = f'with{effect_class.__name__}'
    withEffect.__qualname__ = f'with{effect_class.__name__}'
    withEffect.__doc__ = f'\n    A shortcut PathEffect for applying `.{effect_class.__name__}` and then\n    drawing the original Artist.\n\n    With this class you can use ::\n\n        artist.set_path_effects([patheffects.with{effect_class.__name__}()])\n\n    as a shortcut for ::\n\n        artist.set_path_effects([patheffects.{effect_class.__name__}(),\n                                 patheffects.Normal()])\n    '
    withEffect.draw_path.__doc__ = effect_class.draw_path.__doc__
    return withEffect