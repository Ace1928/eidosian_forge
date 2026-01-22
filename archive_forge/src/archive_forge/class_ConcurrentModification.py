from boto.exception import BotoServerError
class ConcurrentModification(RetriableResponseError):
    """A retriable error can happen when two processes try to modify the
       same data at the same time.
    """