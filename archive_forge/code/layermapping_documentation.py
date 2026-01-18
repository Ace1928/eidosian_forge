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

        Save the contents from the OGR DataSource Layer into the database
        according to the mapping dictionary given at initialization.

        Keyword Parameters:
         verbose:
           If set, information will be printed subsequent to each model save
           executed on the database.

         fid_range:
           May be set with a slice or tuple of (begin, end) feature ID's to map
           from the data source.  In other words, this keyword enables the user
           to selectively import a subset range of features in the geographic
           data source.

         step:
           If set with an integer, transactions will occur at every step
           interval. For example, if step=1000, a commit would occur after
           the 1,000th feature, the 2,000th feature etc.

         progress:
           When this keyword is set, status information will be printed giving
           the number of features processed and successfully saved.  By default,
           progress information will pe printed every 1000 features processed,
           however, this default may be overridden by setting this keyword with an
           integer for the desired interval.

         stream:
           Status information will be written to this file handle.  Defaults to
           using `sys.stdout`, but any object with a `write` method is supported.

         silent:
           By default, non-fatal error notifications are printed to stdout, but
           this keyword may be set to disable these notifications.

         strict:
           Execution of the model mapping will cease upon the first error
           encountered.  The default behavior is to attempt to continue.
        