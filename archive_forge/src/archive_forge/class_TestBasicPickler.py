import numpy as np
from holoviews import Image, Layout
from holoviews.core.io import Deserializer, Pickler, Serializer, Unpickler
from holoviews.element.comparison import assert_element_equal
class TestBasicPickler:
    """
    Test pickler and unpickler using the .hvz format, including
    metadata access.
    """

    def setup_method(self):
        self.image1 = Image(np.array([[1, 2], [4, 5]]))
        self.image2 = Image(np.array([[5, 4], [3, 2]]))

    def test_pickler_save(self, tmp_path) -> None:
        Pickler.save(self.image1, tmp_path / 'test_pickler_save.hvz', info={'info': 'example'}, key={1: 2})

    def test_pickler_save_no_file_extension(self, tmp_path):
        Pickler.save(self.image1, tmp_path / 'test_pickler_save_no_ext', info={'info': 'example'}, key={1: 2})
        assert tmp_path / 'test_pickler_save_no_ext.hvz' in tmp_path.iterdir()

    def test_pickler_save_and_load_data_1(self, tmp_path):
        Pickler.save(self.image1, tmp_path / 'test_pickler_save_and_load_data_1.hvz')
        loaded = Unpickler.load(tmp_path / 'test_pickler_save_and_load_data_1.hvz')
        assert_element_equal(loaded, self.image1)

    def test_pickler_save_and_load_data_2(self, tmp_path):
        Pickler.save(self.image2, tmp_path / 'test_pickler_save_and_load_data_2.hvz')
        loaded = Unpickler.load(tmp_path / 'test_pickler_save_and_load_data_2.hvz')
        assert_element_equal(loaded, self.image2)

    def test_pickler_save_and_load_key(self, tmp_path):
        input_key = {'test_key': 'key_val'}
        Pickler.save(self.image1, tmp_path / 'test_pickler_save_and_load_data.hvz', key=input_key)
        key = Unpickler.key(tmp_path / 'test_pickler_save_and_load_data.hvz')
        assert key == input_key

    def test_pickler_save_and_load_info(self, tmp_path):
        input_info = {'info': 'example'}
        Pickler.save(self.image1, tmp_path / 'test_pickler_save_and_load_data.hvz', info=input_info)
        info = Unpickler.info(tmp_path / 'test_pickler_save_and_load_data.hvz')
        assert info['info'] == input_info['info']

    def test_serialize_deserialize_1(self):
        data, _ = Pickler(self.image1)
        obj = Unpickler(data)
        assert_element_equal(obj, self.image1)

    def test_serialize_deserialize_2(self):
        data, _ = Pickler(self.image2)
        obj = Unpickler(data)
        assert_element_equal(obj, self.image2)