from functools import wraps
import logging
class _QueryLogHandler(logging.Handler):

    def __init__(self, *args, **kwargs):
        self.queries = []
        logging.Handler.__init__(self, *args, **kwargs)

    def emit(self, record):
        if record.name == 'peewee':
            self.queries.append(record)