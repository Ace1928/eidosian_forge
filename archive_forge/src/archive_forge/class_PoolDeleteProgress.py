class PoolDeleteProgress(object):

    def __init__(self, task_complete=False):
        self.pool = {'delete_called': task_complete, 'deleted': task_complete}
        self.vip = {'delete_called': task_complete, 'deleted': task_complete}