from unittest import SkipTest
import numpy as np
from holoviews.core.data import Dataset, concat
from geoviews.data.iris import coord_to_dimension
from holoviews.core.spaces import HoloMap
from holoviews.element import Image
from holoviews.tests.core.data.test_imageinterface import BaseImageElementInterfaceTests
from holoviews.tests.core.data.test_gridinterface import BaseGridInterfaceTests
def test_select_multi_slice2(self):
    cube = Dataset(self.cube)
    self.assertEqual(cube.select(longitude={0, 2}, latitude={0, 2}).data.data, np.array([[5, 7]], dtype=np.int32))