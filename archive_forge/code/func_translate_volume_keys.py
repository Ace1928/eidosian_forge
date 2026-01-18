import sys
import time
import prettytable
from cinderclient import exceptions
from cinderclient import utils
def translate_volume_keys(collection):
    convert = [('volumeType', 'volume_type'), ('os-vol-tenant-attr:tenant_id', 'tenant_id')]
    translate_keys(collection, convert)