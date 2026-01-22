from django.core.exceptions import FieldError
from django.db.models.sql.constants import CURSOR, GET_ITERATOR_CHUNK_SIZE, NO_RESULTS
from django.db.models.sql.query import Query
def _setup_query(self):
    """
        Run on initialization and at the end of chaining. Any attributes that
        would normally be set in __init__() should go here instead.
        """
    self.values = []
    self.related_ids = None
    self.related_updates = {}