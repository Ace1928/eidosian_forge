import datetime
from io import BytesIO
from pathlib import Path
import xml.etree.ElementTree
import xml.parsers.expat
import pytest
import numpy as np
import matplotlib as mpl
from matplotlib.figure import Figure
from matplotlib.text import Text
import matplotlib.pyplot as plt
from matplotlib.testing.decorators import check_figures_equal, image_comparison
from matplotlib.testing._markers import needs_usetex
from matplotlib import font_manager as fm
from matplotlib.offsetbox import (OffsetImage, AnnotationBbox)
@pytest.mark.parametrize('metadata,error,message', [({'Date': 1}, TypeError, 'Invalid type for Date metadata. Expected str'), ({'Date': [1]}, TypeError, 'Invalid type for Date metadata. Expected iterable'), ({'Keywords': 1}, TypeError, 'Invalid type for Keywords metadata. Expected str'), ({'Keywords': [1]}, TypeError, 'Invalid type for Keywords metadata. Expected iterable'), ({'Creator': 1}, TypeError, 'Invalid type for Creator metadata. Expected str'), ({'Creator': [1]}, TypeError, 'Invalid type for Creator metadata. Expected iterable'), ({'Title': 1}, TypeError, 'Invalid type for Title metadata. Expected str'), ({'Format': 1}, TypeError, 'Invalid type for Format metadata. Expected str'), ({'Foo': 'Bar'}, ValueError, 'Unknown metadata key')])
def test_svg_incorrect_metadata(metadata, error, message):
    with pytest.raises(error, match=message), BytesIO() as fd:
        fig = plt.figure()
        fig.savefig(fd, format='svg', metadata=metadata)