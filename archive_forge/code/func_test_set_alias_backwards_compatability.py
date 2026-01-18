import time
from tests.compat import unittest
from boto.route53.connection import Route53Connection
from boto.route53.record import ResourceRecordSets
from boto.route53.exception import DNSServerError
def test_set_alias_backwards_compatability(self):
    base_record = dict(name='alias.%s.' % self.base_domain, type='A', identifier='boto:TestRoute53AliasResourceRecordSets')
    rrs = ResourceRecordSets(self.conn, self.zone.id)
    new = rrs.add_change(action='UPSERT', **base_record)
    new.set_alias(self.zone.id, 'target.%s' % self.base_domain)
    rrs.commit()
    rrs = ResourceRecordSets(self.conn, self.zone.id)
    delete = rrs.add_change(action='DELETE', **base_record)
    delete.set_alias(self.zone.id, 'target.%s' % self.base_domain)
    rrs.commit()