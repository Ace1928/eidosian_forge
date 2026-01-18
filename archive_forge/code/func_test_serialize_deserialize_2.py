import numpy as np
from holoviews import Image, Layout
from holoviews.core.io import Deserializer, Pickler, Serializer, Unpickler
from holoviews.element.comparison import assert_element_equal
def test_serialize_deserialize_2(self):
    data, _ = Pickler(self.image2)
    obj = Unpickler(data)
    assert_element_equal(obj, self.image2)