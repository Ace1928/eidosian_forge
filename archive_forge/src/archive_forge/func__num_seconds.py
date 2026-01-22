from datetime import datetime
from django.conf import settings
from django.utils.crypto import constant_time_compare, salted_hmac
from django.utils.http import base36_to_int, int_to_base36
def _num_seconds(self, dt):
    return int((dt - datetime(2001, 1, 1)).total_seconds())