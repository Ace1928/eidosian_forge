import threading
from tensorboard import errors
def _checkBeginEndIndices(self, begin, end, total_count):
    if begin < 0:
        raise errors.InvalidArgumentError('Invalid begin index (%d)' % begin)
    if end > total_count:
        raise errors.InvalidArgumentError('end index (%d) out of bounds (%d)' % (end, total_count))
    if end >= 0 and end < begin:
        raise errors.InvalidArgumentError('end index (%d) is unexpectedly less than begin index (%d)' % (end, begin))
    if end < 0:
        end = total_count
    return end