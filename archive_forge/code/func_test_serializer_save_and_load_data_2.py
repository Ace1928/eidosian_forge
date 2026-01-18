import numpy as np
from holoviews import Image, Layout
from holoviews.core.io import Deserializer, Pickler, Serializer, Unpickler
from holoviews.element.comparison import assert_element_equal
def test_serializer_save_and_load_data_2(self, tmp_path):
    Serializer.save(self.image2, tmp_path / 'test_serializer_save_and_load_data_2.pkl')
    loaded = Deserializer.load(tmp_path / 'test_serializer_save_and_load_data_2.pkl')
    assert_element_equal(loaded, self.image2)