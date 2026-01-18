import contextlib
import string
import threading
import time
from oslo_utils import timeutils
import redis
from taskflow import exceptions
from taskflow.listeners import capturing
from taskflow.persistence.backends import impl_memory
from taskflow import retry
from taskflow import task
from taskflow.types import failure
from taskflow.utils import kazoo_utils
from taskflow.utils import redis_utils
def zookeeper_available(min_version, timeout=3):
    client = kazoo_utils.make_client(ZK_TEST_CONFIG.copy())
    try:
        client.start(timeout=float(timeout))
        if min_version:
            zk_ver = client.server_version()
            if zk_ver >= min_version:
                return True
            else:
                return False
        else:
            return True
    except Exception:
        return False
    finally:
        kazoo_utils.finalize_client(client)