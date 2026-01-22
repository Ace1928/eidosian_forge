import errno
import os
from stat import S_ISDIR
from wandb_watchdog.utils import stat as default_stat
Allow subtracting a DirectorySnapshot object instance from
        another.

        :returns:
            A :class:`DirectorySnapshotDiff` object.
        