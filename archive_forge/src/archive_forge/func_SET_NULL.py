from collections import Counter, defaultdict
from functools import partial, reduce
from itertools import chain
from operator import attrgetter, or_
from django.db import IntegrityError, connections, models, transaction
from django.db.models import query_utils, signals, sql
def SET_NULL(collector, field, sub_objs, using):
    collector.add_field_update(field, None, sub_objs)