import zoneinfo
from datetime import datetime
from datetime import timezone as datetime_timezone
from datetime import tzinfo
from django.template import Library, Node, TemplateSyntaxError
from django.utils import timezone
@register.filter
def utc(value):
    """
    Convert a datetime to UTC.
    """
    return do_timezone(value, datetime_timezone.utc)