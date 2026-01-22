import sys
from datetime import datetime
from threading import Thread
import Queue
from boto.utils import RequestHook
from boto.compat import long_type
def _request_log_worker(self):
    while True:
        try:
            item = self.request_log_queue.get(True)
            self.request_log_file.write(item)
            self.request_log_file.flush()
            self.request_log_queue.task_done()
        except:
            import traceback
            traceback.print_exc(file=sys.stdout)