import bisect
from tensorboard.backend.event_processing import io_wrapper
from tensorboard.compat import tf
from tensorboard.util import io_util
from tensorboard.util import tb_logging
Returns whether the path has had an out-of-order write.