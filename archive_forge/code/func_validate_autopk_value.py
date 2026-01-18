import uuid
from django.conf import settings
from django.db.backends.base.operations import BaseDatabaseOperations
from django.db.backends.utils import split_tzname_delta
from django.db.models import Exists, ExpressionWrapper, Lookup
from django.db.models.constants import OnConflict
from django.utils import timezone
from django.utils.encoding import force_str
from django.utils.regex_helper import _lazy_re_compile
def validate_autopk_value(self, value):
    if value == 0 and (not self.connection.features.allows_auto_pk_0):
        raise ValueError('The database backend does not accept 0 as a value for AutoField.')
    return value