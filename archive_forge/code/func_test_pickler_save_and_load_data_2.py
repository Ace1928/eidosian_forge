import numpy as np
from holoviews import Image, Layout
from holoviews.core.io import Deserializer, Pickler, Serializer, Unpickler
from holoviews.element.comparison import assert_element_equal
def test_pickler_save_and_load_data_2(self, tmp_path):
    Pickler.save(self.image2, tmp_path / 'test_pickler_save_and_load_data_2.hvz')
    loaded = Unpickler.load(tmp_path / 'test_pickler_save_and_load_data_2.hvz')
    assert_element_equal(loaded, self.image2)