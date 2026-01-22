from django.core.exceptions import FieldError
from django.db.models.sql.constants import CURSOR, GET_ITERATOR_CHUNK_SIZE, NO_RESULTS
from django.db.models.sql.query import Query
def add_related_update(self, model, field, value):
    """
        Add (name, value) to an update query for an ancestor model.

        Update are coalesced so that only one update query per ancestor is run.
        """
    self.related_updates.setdefault(model, []).append((field, None, value))