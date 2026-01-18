import sys
import time
import prettytable
from cinderclient import exceptions
from cinderclient import utils
def translate_availability_zone_keys(collection):
    convert = [('zoneName', 'name'), ('zoneState', 'status')]
    translate_keys(collection, convert)