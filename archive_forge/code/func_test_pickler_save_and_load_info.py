import numpy as np
from holoviews import Image, Layout
from holoviews.core.io import Deserializer, Pickler, Serializer, Unpickler
from holoviews.element.comparison import assert_element_equal
def test_pickler_save_and_load_info(self, tmp_path):
    input_info = {'info': 'example'}
    Pickler.save(self.image1, tmp_path / 'test_pickler_save_and_load_data.hvz', info=input_info)
    info = Unpickler.info(tmp_path / 'test_pickler_save_and_load_data.hvz')
    assert info['info'] == input_info['info']