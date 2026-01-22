import boto
from boto.sdb.db.property import StringProperty, DateTimeProperty, IntegerProperty
from boto.sdb.db.model import Model
import datetime, subprocess, time
from boto.compat import StringIO
class TaskPoller(object):

    def __init__(self, queue_name):
        self.sqs = boto.connect_sqs()
        self.queue = self.sqs.lookup(queue_name)

    def poll(self, wait=60, vtimeout=60):
        while True:
            m = self.queue.read(vtimeout)
            if m:
                task = Task.get_by_id(m.get_body())
                if task:
                    if not task.message_id or m.id == task.message_id:
                        boto.log.info('Task[%s] - read message %s' % (task.name, m.id))
                        task.run(m, vtimeout)
                    else:
                        boto.log.info('Task[%s] - found extraneous message, ignoring' % task.name)
            else:
                time.sleep(wait)