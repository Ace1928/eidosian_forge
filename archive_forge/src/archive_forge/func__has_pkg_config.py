import gc as _gc
import importlib as _importlib
import os as _os
import platform as _platform
import sys as _sys
import warnings as _warnings
import pyarrow.lib as _lib
from pyarrow.lib import (BuildInfo, RuntimeInfo, set_timezone_db_path,
from pyarrow.lib import (null, bool_,
from pyarrow.lib import (Buffer, ResizableBuffer, foreign_buffer, py_buffer,
from pyarrow.lib import (MemoryPool, LoggingMemoryPool, ProxyMemoryPool,
from pyarrow.lib import (NativeFile, PythonFile,
from pyarrow._hdfsio import HdfsFile, have_libhdfs
from pyarrow.lib import (ChunkedArray, RecordBatch, Table, table,
from pyarrow.lib import (ArrowCancelled,
import pyarrow.hdfs as hdfs
from pyarrow.ipc import serialize_pandas, deserialize_pandas
import pyarrow.ipc as ipc
import pyarrow.types as types
from pyarrow.filesystem import FileSystem as _FileSystem
from pyarrow.filesystem import LocalFileSystem as _LocalFileSystem
from pyarrow.hdfs import HadoopFileSystem as _HadoopFileSystem
from pyarrow.util import _deprecate_api, _deprecate_class
from pyarrow.ipc import (Message, MessageReader, MetadataVersion,
def _has_pkg_config(pkgname):
    import subprocess
    try:
        return subprocess.call([_get_pkg_config_executable(), '--exists', pkgname]) == 0
    except FileNotFoundError:
        return False