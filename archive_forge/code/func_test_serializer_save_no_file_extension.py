import numpy as np
from holoviews import Image, Layout
from holoviews.core.io import Deserializer, Pickler, Serializer, Unpickler
from holoviews.element.comparison import assert_element_equal
def test_serializer_save_no_file_extension(self, tmp_path):
    Serializer.save(self.image1, tmp_path / 'test_serializer_save_no_ext', info={'info': 'example'}, key={1: 2})
    assert tmp_path / 'test_serializer_save_no_ext.pkl' in tmp_path.iterdir()