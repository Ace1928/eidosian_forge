from datetime import datetime
from django.conf import settings
from django.db.models.expressions import Func
from django.db.models.fields import (
from django.db.models.lookups import (
from django.utils import timezone
class TruncWeek(TruncBase):
    """Truncate to midnight on the Monday of the week."""
    kind = 'week'