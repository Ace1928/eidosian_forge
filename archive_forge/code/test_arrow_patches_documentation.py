import pytest
import platform
import matplotlib.pyplot as plt
from matplotlib.testing.decorators import image_comparison
import matplotlib.patches as mpatches

    As test_fancyarrow_dpi_cor_100dpi, but exports @ 200 DPI. The relative size
    of the arrow head should be the same.
    