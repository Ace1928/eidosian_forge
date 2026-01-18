import xml.sax
import datetime
import itertools
from boto import handler
from boto import config
from boto.mturk.price import Price
import boto.mturk.notification
from boto.connection import AWSQueryConnection
from boto.exception import EC2ResponseError
from boto.resultset import ResultSet
from boto.mturk.question import QuestionForm, ExternalQuestion, HTMLQuestion
def unblock_worker(self, worker_id, reason):
    """
        Unblock a worker from working on my tasks.
        """
    params = {'WorkerId': worker_id, 'Reason': reason}
    return self._process_request('UnblockWorker', params)