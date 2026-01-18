import numpy
from thinc.api import HashEmbed
def test_seed_changes_bucket():
    model1 = HashEmbed(64, 1000, seed=2).initialize()
    model2 = HashEmbed(64, 1000, seed=1).initialize()
    arr = numpy.ones((1,), dtype='uint64')
    vector1 = model1.predict(arr)
    vector2 = model2.predict(arr)
    assert vector1.sum() != vector2.sum()