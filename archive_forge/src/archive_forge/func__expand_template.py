import logging
import re
from boto.vendored.regions.exceptions import NoRegionError
def _expand_template(self, partition, template, service_name, endpoint_name):
    return template.format(service=service_name, region=endpoint_name, dnsSuffix=partition['dnsSuffix'])