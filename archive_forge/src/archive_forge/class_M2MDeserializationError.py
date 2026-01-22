from io import StringIO
from django.core.exceptions import ObjectDoesNotExist
from django.db import models
class M2MDeserializationError(Exception):
    """Something bad happened during deserialization of a ManyToManyField."""

    def __init__(self, original_exc, pk):
        self.original_exc = original_exc
        self.pk = pk