import numpy as np
from holoviews import Image, Layout
from holoviews.core.io import Deserializer, Pickler, Serializer, Unpickler
from holoviews.element.comparison import assert_element_equal
class TestPicklerAdvanced:
    """
    Test advanced pickler and unpickler functionality supported by the
    .hvz format.
    """

    def setup_method(self):
        self.image1 = Image(np.array([[1, 2], [4, 5]]))
        self.image2 = Image(np.array([[5, 4], [3, 2]]))

    def test_pickler_save_layout(self, tmp_path):
        Pickler.save(self.image1 + self.image2, tmp_path / 'test_pickler_save_layout', info={'info': 'example'}, key={1: 2})

    def test_pickler_save_load_layout(self, tmp_path):
        Pickler.save(self.image1 + self.image2, tmp_path / 'test_pickler_save_load_layout', info={'info': 'example'}, key={1: 2})
        loaded = Unpickler.load(tmp_path / 'test_pickler_save_load_layout.hvz')
        assert_element_equal(loaded, self.image1 + self.image2)

    def test_pickler_save_load_layout_entries(self, tmp_path):
        Pickler.save(self.image1 + self.image2, tmp_path / 'test_pickler_save_load_layout_entries', info={'info': 'example'}, key={1: 2})
        entries = Unpickler.entries(tmp_path / 'test_pickler_save_load_layout_entries.hvz')
        assert entries == ['Image.I', 'Image.II']

    def test_pickler_save_load_layout_entry1(self, tmp_path):
        Pickler.save(self.image1 + self.image2, tmp_path / 'test_pickler_save_load_layout_entry1', info={'info': 'example'}, key={1: 2})
        entries = Unpickler.entries(tmp_path / 'test_pickler_save_load_layout_entry1.hvz')
        assert 'Image.I' in entries, "Entry 'Image.I' missing"
        loaded = Unpickler.load(tmp_path / 'test_pickler_save_load_layout_entry1.hvz', entries=['Image.I'])
        assert_element_equal(loaded, self.image1)

    def test_pickler_save_load_layout_entry2(self, tmp_path):
        Pickler.save(self.image1 + self.image2, tmp_path / 'test_pickler_save_load_layout_entry2', info={'info': 'example'}, key={1: 2})
        entries = Unpickler.entries(tmp_path / 'test_pickler_save_load_layout_entry2.hvz')
        assert 'Image.II' in entries, "Entry 'Image.II' missing"
        loaded = Unpickler.load(tmp_path / 'test_pickler_save_load_layout_entry2.hvz', entries=['Image.II'])
        assert_element_equal(loaded, self.image2)

    def test_pickler_save_load_single_layout(self, tmp_path):
        single_layout = Layout([self.image1])
        Pickler.save(single_layout, tmp_path / 'test_pickler_save_load_single_layout', info={'info': 'example'}, key={1: 2})
        entries = Unpickler.entries(tmp_path / 'test_pickler_save_load_single_layout.hvz')
        assert entries == ['Image.I(L)']
        loaded = Unpickler.load(tmp_path / 'test_pickler_save_load_single_layout.hvz', entries=['Image.I(L)'])
        assert_element_equal(single_layout, loaded)