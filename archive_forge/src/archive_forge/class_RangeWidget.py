from django import forms
from django.core import exceptions
from django.db.backends.postgresql.psycopg_any import (
from django.forms.widgets import HiddenInput, MultiWidget
from django.utils.translation import gettext_lazy as _
class RangeWidget(MultiWidget):

    def __init__(self, base_widget, attrs=None):
        widgets = (base_widget, base_widget)
        super().__init__(widgets, attrs)

    def decompress(self, value):
        if value:
            return (value.lower, value.upper)
        return (None, None)