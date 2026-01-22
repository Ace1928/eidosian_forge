import collections
from unittest import mock
from openstack.network.v2 import firewall_group as fw_group
from openstack.network.v2 import firewall_policy as fw_policy
from openstack.network.v2 import firewall_rule as fw_rule
from oslo_utils import uuidutils
class FakeFWaaS(object):

    def create(self, attrs={}):
        """Create a fake fwaas resources

        :param Dictionary attrs:
            A dictionary with all attributes
        :return:
            A OrderedDict faking the fwaas resource
        """
        self.ordered.update(attrs)
        if 'FirewallGroup' == self.__class__.__name__:
            return fw_group.FirewallGroup(**self.ordered)
        if 'FirewallPolicy' == self.__class__.__name__:
            return fw_policy.FirewallPolicy(**self.ordered)
        if 'FirewallRule' == self.__class__.__name__:
            fw_r = fw_rule.FirewallRule(**self.ordered)
            protocol = fw_r['protocol'].upper() if fw_r['protocol'] else 'ANY'
            src_ip = str(fw_r['source_ip_address']).lower()
            src_port = '(' + str(fw_r['source_port']).lower() + ')'
            dst_ip = str(fw_r['destination_ip_address']).lower()
            dst_port = '(' + str(fw_r['destination_port']).lower() + ')'
            src = 'source(port): ' + src_ip + src_port
            dst = 'dest(port): ' + dst_ip + dst_port
            action = fw_r['action'] if fw_r.get('action') else 'no-action'
            fw_r['summary'] = ',\n '.join([protocol, src, dst, action])
            return fw_r

    def bulk_create(self, attrs=None, count=2):
        """Create multiple fake fwaas resources

        :param Dictionary attrs:
            A dictionary with all attributes
        :param int count:
            The number of fwaas resources to fake
        :return:
            A list of dictionaries faking the fwaas resources
        """
        return [self.create(attrs=attrs) for i in range(0, count)]

    def get(self, attrs=None, count=2):
        """Create multiple fake fwaas resources

        :param Dictionary attrs:
            A dictionary with all attributes
        :param int count:
            The number of fwaas resources to fake
        :return:
            A list of dictionaries faking the fwaas resource
        """
        if attrs is None:
            self.attrs = self.bulk_create(count=count)
        return mock.Mock(side_effect=attrs)