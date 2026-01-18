from ..sage_helper import _within_sage
import math
def translations_from_cusp_shape_and_area(cusp_shape, cusp_area, kernel_convention=False):
    """
    Unfortunately, the short_slopes_from_translations uses the convention
    that the longitude translation is real whereas the SnapPea kernel and
    M.cusp_translations uses the convention that the meridian translation
    is real. Thus, we have a flag here.

    Ideally, we would rewrite below short_slopes_from_translations to
    use the same convention.
    """
    if kernel_convention:
        inv_cusp_shape = 1 / cusp_shape.conjugate()
        scale = sqrt(cusp_area / _imag(inv_cusp_shape))
        return (scale * inv_cusp_shape, scale)
    else:
        scale = sqrt(cusp_area / _imag(cusp_shape))
        return (scale, cusp_shape * scale)