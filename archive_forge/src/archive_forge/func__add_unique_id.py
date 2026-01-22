import collections
import uuid
from oslo_config import cfg
from oslo_messaging._drivers import common as rpc_common
def _add_unique_id(msg):
    """Add unique_id for checking duplicate messages."""
    unique_id = uuid.uuid4().hex
    msg.update({UNIQUE_ID: unique_id})