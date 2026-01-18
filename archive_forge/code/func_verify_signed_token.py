from __future__ import absolute_import, unicode_literals
import collections
import datetime
import logging
import re
import sys
import time
def verify_signed_token(public_pem, token):
    import jwt
    return jwt.decode(token, public_pem, algorithms=['RS256'])