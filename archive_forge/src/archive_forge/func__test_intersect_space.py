import minerl.herobraine.envs as envs
from minerl.herobraine.wrappers.util import intersect_space
def _test_intersect_space(space, sample):
    intersected = intersect_space(space, sample)
    assert intersected in space