import time
from tests.compat import unittest
from boto.route53.connection import Route53Connection
from boto.route53.record import ResourceRecordSets
from boto.route53.exception import DNSServerError
class TestRoute53AliasResourceRecordSets(unittest.TestCase):
    route53 = True

    def setUp(self):
        super(TestRoute53AliasResourceRecordSets, self).setUp()
        self.conn = Route53Connection()
        self.base_domain = 'boto-test-%s.com' % str(int(time.time()))
        self.zone = self.conn.create_zone(self.base_domain)
        self.zone.add_a('target.%s' % self.base_domain, '102.11.23.1')

    def tearDown(self):
        self.zone.delete_a('target.%s' % self.base_domain)
        self.zone.delete()
        super(TestRoute53AliasResourceRecordSets, self).tearDown()

    def test_incomplete_add_alias_failure(self):
        base_record = dict(name='alias.%s.' % self.base_domain, type='A', alias_dns_name='target.%s' % self.base_domain, alias_hosted_zone_id=self.zone.id, identifier='boto:TestRoute53AliasResourceRecordSets')
        rrs = ResourceRecordSets(self.conn, self.zone.id)
        rrs.add_change(action='UPSERT', **base_record)
        try:
            self.assertRaises(DNSServerError, rrs.commit)
        except:
            rrs = ResourceRecordSets(self.conn, self.zone.id)
            rrs.add_change(action='DELETE', **base_record)
            rrs.commit()
            raise

    def test_add_alias(self):
        base_record = dict(name='alias.%s.' % self.base_domain, type='A', alias_evaluate_target_health=False, alias_dns_name='target.%s' % self.base_domain, alias_hosted_zone_id=self.zone.id, identifier='boto:TestRoute53AliasResourceRecordSets')
        rrs = ResourceRecordSets(self.conn, self.zone.id)
        rrs.add_change(action='UPSERT', **base_record)
        rrs.commit()
        rrs = ResourceRecordSets(self.conn, self.zone.id)
        rrs.add_change(action='DELETE', **base_record)
        rrs.commit()

    def test_set_alias(self):
        base_record = dict(name='alias.%s.' % self.base_domain, type='A', identifier='boto:TestRoute53AliasResourceRecordSets')
        rrs = ResourceRecordSets(self.conn, self.zone.id)
        new = rrs.add_change(action='UPSERT', **base_record)
        new.set_alias(self.zone.id, 'target.%s' % self.base_domain, False)
        rrs.commit()
        rrs = ResourceRecordSets(self.conn, self.zone.id)
        delete = rrs.add_change(action='DELETE', **base_record)
        delete.set_alias(self.zone.id, 'target.%s' % self.base_domain, False)
        rrs.commit()

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