from collections import Counter
from django.conf import settings
from . import Error, Tags, Warning, register
def E006(name):
    return Error('The {} setting must end with a slash.'.format(name), id='urls.E006')