import functools
import threading
from google.api_core import exceptions
from google.api_core import protobuf_helpers
from google.api_core.future import async_future
from google.longrunning import operations_pb2
from google.rpc import code_pb2
True if the operation was cancelled.