import re
from datetime import date, datetime, timezone
from decimal import Decimal
from django import template
from django.template import defaultfilters
from django.utils.formats import number_format
from django.utils.safestring import mark_safe
from django.utils.timezone import is_aware
from django.utils.translation import gettext as _
from django.utils.translation import (

    For date and time values show how many seconds, minutes, or hours ago
    compared to current timestamp return representing string.
    