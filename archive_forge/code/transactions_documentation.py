from __future__ import absolute_import
import re
from sentry_sdk._types import TYPE_CHECKING
from django import VERSION as DJANGO_VERSION

        Clean up urlpattern regexes into something readable by humans:

        From:
        > "^(?P<sport_slug>\w+)/athletes/(?P<athlete_slug>\w+)/$"

        To:
        > "{sport_slug}/athletes/{athlete_slug}/"
        