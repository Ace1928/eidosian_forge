import io
import logging
import random
import fixtures
from openstackclient import shell
from oslotest import base
from placement.tests.functional.fixtures import capture
from placement.tests.functional.fixtures import placement
import simplejson as json
def _allocation_candidates_option(self, resources=None, required=None, forbidden=None, aggregate_uuids=None, member_of=None):
    opt = ''
    if resources:
        opt += ' ' + ' '.join(('--resource %s' % resource for resource in resources))
    if required is not None:
        opt += ''.join([' --required %s' % t for t in required])
    if forbidden:
        opt += ' ' + ' '.join(('--forbidden %s' % f for f in forbidden))
    if aggregate_uuids:
        opt += ' ' + ' '.join(('--aggregate-uuid %s' % a for a in aggregate_uuids))
    if member_of:
        opt += ' ' + ' '.join(['--member-of %s' % m for m in member_of])
    return opt