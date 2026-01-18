import numpy as np
from holoviews import Image, Layout
from holoviews.core.io import Deserializer, Pickler, Serializer, Unpickler
from holoviews.element.comparison import assert_element_equal
def test_pickler_save_load_layout_entries(self, tmp_path):
    Pickler.save(self.image1 + self.image2, tmp_path / 'test_pickler_save_load_layout_entries', info={'info': 'example'}, key={1: 2})
    entries = Unpickler.entries(tmp_path / 'test_pickler_save_load_layout_entries.hvz')
    assert entries == ['Image.I', 'Image.II']