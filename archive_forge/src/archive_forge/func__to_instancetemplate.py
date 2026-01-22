import sys
import time
import datetime
import itertools
from libcloud.pricing import get_pricing
from libcloud.common.base import LazyObject
from libcloud.common.types import LibcloudError
from libcloud.compute.base import (
from libcloud.common.google import (
from libcloud.compute.types import NodeState
from libcloud.utils.iso8601 import parse_date
from libcloud.compute.providers import Provider
def _to_instancetemplate(self, instancetemplate):
    """
        Return a Instance Template object from the JSON-response.

        :param  instancetemplate: dictionary describing the Instance
                                  Template.
        :type   instancetemplate: ``dict``

        :return: Instance Template object.
        :rtype:  :class:`GCEInstanceTemplate`
        """
    extra = {}
    extra['selfLink'] = instancetemplate.get('selfLink')
    extra['description'] = instancetemplate.get('description')
    extra['properties'] = instancetemplate.get('properties')
    return GCEInstanceTemplate(id=instancetemplate['id'], name=instancetemplate['name'], driver=self, extra=extra)