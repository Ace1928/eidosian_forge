import sys
from decimal import Decimal
from decimal import InvalidOperation as DecimalInvalidOperation
from pathlib import Path
from django.contrib.gis.db.models import GeometryField
from django.contrib.gis.gdal import (
from django.contrib.gis.gdal.field import (
from django.core.exceptions import FieldDoesNotExist, ObjectDoesNotExist
from django.db import connections, models, router, transaction
from django.utils.encoding import force_str
def verify_ogr_field(self, ogr_field, model_field):
    """
        Verify if the OGR Field contents are acceptable to the model field. If
        they are, return the verified value, otherwise raise an exception.
        """
    if isinstance(ogr_field, OFTString) and isinstance(model_field, (models.CharField, models.TextField)):
        if self.encoding and ogr_field.value is not None:
            val = force_str(ogr_field.value, self.encoding)
        else:
            val = ogr_field.value
        if model_field.max_length and val is not None and (len(val) > model_field.max_length):
            raise InvalidString('%s model field maximum string length is %s, given %s characters.' % (model_field.name, model_field.max_length, len(val)))
    elif isinstance(ogr_field, OFTReal) and isinstance(model_field, models.DecimalField):
        try:
            d = Decimal(str(ogr_field.value))
        except DecimalInvalidOperation:
            raise InvalidDecimal('Could not construct decimal from: %s' % ogr_field.value)
        dtup = d.as_tuple()
        digits = dtup[1]
        d_idx = dtup[2]
        max_prec = model_field.max_digits - model_field.decimal_places
        if d_idx < 0:
            n_prec = len(digits[:d_idx])
        else:
            n_prec = len(digits) + d_idx
        if n_prec > max_prec:
            raise InvalidDecimal('A DecimalField with max_digits %d, decimal_places %d must round to an absolute value less than 10^%d.' % (model_field.max_digits, model_field.decimal_places, max_prec))
        val = d
    elif isinstance(ogr_field, (OFTReal, OFTString)) and isinstance(model_field, models.IntegerField):
        try:
            val = int(ogr_field.value)
        except ValueError:
            raise InvalidInteger('Could not construct integer from: %s' % ogr_field.value)
    else:
        val = ogr_field.value
    return val