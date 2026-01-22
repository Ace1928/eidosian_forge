from collections import Counter, defaultdict
from functools import partial, reduce
from itertools import chain
from operator import attrgetter, or_
from django.db import IntegrityError, connections, models, transaction
from django.db.models import query_utils, signals, sql
class RestrictedError(IntegrityError):

    def __init__(self, msg, restricted_objects):
        self.restricted_objects = restricted_objects
        super().__init__(msg, restricted_objects)