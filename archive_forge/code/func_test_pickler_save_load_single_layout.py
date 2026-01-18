import numpy as np
from holoviews import Image, Layout
from holoviews.core.io import Deserializer, Pickler, Serializer, Unpickler
from holoviews.element.comparison import assert_element_equal
def test_pickler_save_load_single_layout(self, tmp_path):
    single_layout = Layout([self.image1])
    Pickler.save(single_layout, tmp_path / 'test_pickler_save_load_single_layout', info={'info': 'example'}, key={1: 2})
    entries = Unpickler.entries(tmp_path / 'test_pickler_save_load_single_layout.hvz')
    assert entries == ['Image.I(L)']
    loaded = Unpickler.load(tmp_path / 'test_pickler_save_load_single_layout.hvz', entries=['Image.I(L)'])
    assert_element_equal(single_layout, loaded)