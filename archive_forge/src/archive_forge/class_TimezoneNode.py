import zoneinfo
from datetime import datetime
from datetime import timezone as datetime_timezone
from datetime import tzinfo
from django.template import Library, Node, TemplateSyntaxError
from django.utils import timezone
class TimezoneNode(Node):
    """
    Template node class used by ``timezone_tag``.
    """

    def __init__(self, nodelist, tz):
        self.nodelist = nodelist
        self.tz = tz

    def render(self, context):
        with timezone.override(self.tz.resolve(context)):
            output = self.nodelist.render(context)
        return output