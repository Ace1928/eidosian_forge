from enum import Enum, IntEnum
class _OverviewResampling(IntEnum):
    """Available Overview resampling algorithms.

    The first 8, 'nearest', 'bilinear', 'cubic', 'cubic_spline',
    'lanczos', 'average', 'mode', and 'gauss', are available for making
    dataset overviews.

    'nearest', 'bilinear', 'cubic', 'cubic_spline', 'lanczos',
    'average', 'mode' are always available (GDAL >= 1.10).

    'rms' is only supported in GDAL >= 3.3.

    """
    nearest = 0
    bilinear = 1
    cubic = 2
    cubic_spline = 3
    lanczos = 4
    average = 5
    mode = 6
    gauss = 7
    rms = 14