from django.template import Library, Node, TemplateSyntaxError
from django.utils import formats

    Force or prevents localization of values.

    Sample usage::

        {% localize off %}
            var pi = {{ 3.1415 }};
        {% endlocalize %}
    