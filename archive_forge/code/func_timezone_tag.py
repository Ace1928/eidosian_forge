import zoneinfo
from datetime import datetime
from datetime import timezone as datetime_timezone
from datetime import tzinfo
from django.template import Library, Node, TemplateSyntaxError
from django.utils import timezone
@register.tag('timezone')
def timezone_tag(parser, token):
    """
    Enable a given time zone just for this block.

    The ``timezone`` argument must be an instance of a ``tzinfo`` subclass, a
    time zone name, or ``None``. If it is ``None``, the default time zone is
    used within the block.

    Sample usage::

        {% timezone "Europe/Paris" %}
            It is {{ now }} in Paris.
        {% endtimezone %}
    """
    bits = token.split_contents()
    if len(bits) != 2:
        raise TemplateSyntaxError("'%s' takes one argument (timezone)" % bits[0])
    tz = parser.compile_filter(bits[1])
    nodelist = parser.parse(('endtimezone',))
    parser.delete_first_token()
    return TimezoneNode(nodelist, tz)